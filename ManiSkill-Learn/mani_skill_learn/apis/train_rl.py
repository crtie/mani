import itertools
import os
import os.path as osp
import time
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from mani_skill_learn.env.builder import build_replay
import numpy as np

from mani_skill_learn.env import ReplayMemory
from mani_skill_learn.env import save_eval_statistics
from mani_skill_learn.utils.data import dict_to_str, get_shape, is_seq_of
from mani_skill_learn.utils.meta import get_logger, get_total_memory, td_format
from mani_skill_learn.utils.data import to_torch,to_np,merge_dict
from mani_skill_learn.utils.data import dict_to_str, get_shape, is_seq_of, concat_list_of_array
from mani_skill_learn.utils.torch import TensorboardLogger, save_checkpoint
from mani_skill_learn.utils.math import split_num
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#writer = SummaryWriter("logs")
import chamfer3D.dist_chamfer_3D

class EpisodicStatistics:
    def __init__(self, num_procs):
        self.num_procs = num_procs
        self.current_lens = np.zeros(num_procs)
        self.current_rewards = np.zeros(num_procs)
        self.history_rewards = np.zeros(num_procs)
        self.history_lens = np.zeros(num_procs)
        self.history_counts = np.zeros(num_procs)

    def push(self, rewards, dones):
        n, running_steps = split_num(len(dones), self.num_procs)
        j = 0
        for i in range(n):
            for _ in range(running_steps[i]):
                self.current_lens[i] += 1
                self.current_rewards[i] += rewards[j]
                if dones[j]:
                    self.history_rewards[i] += self.current_rewards[i]
                    self.history_lens[i] += self.current_lens[i]
                    self.history_counts[i] += 1
                    self.current_rewards[i] = 0
                    self.current_lens[i] = 0
                j += 1

    def reset_history(self):
        self.history_lens *= 0
        self.history_rewards *= 0
        self.history_counts *= 0

    def reset_current(self):
        self.current_rewards *= 0
        self.current_lens *= 0

    def get_mean(self):
        num_episode = np.clip(np.sum(self.history_counts), a_min=1E-5, a_max=1E10)
        return np.sum(self.history_lens) / num_episode, np.sum(self.history_rewards) / num_episode

    def print_current(self):
        print(self.current_lens, self.current_rewards)

    def print_history(self):
        print(self.history_lens, self.history_rewards, self.history_counts)


class EveryNSteps:
    def __init__(self, interval=None):
        self.interval = interval
        self.next_value = interval

    def reset(self):
        self.next_value = self.interval

    def check(self, x):
        if self.interval is None:
            return False
        sign = False
        while x >= self.next_value:
            self.next_value += self.interval
            sign = True
        return sign

    def standard(self, x):
        return int(x // self.interval) * self.interval



def train_rl(agent, rollout, evaluator, env_cfg, replay_env ,tmp_replay , replay_model ,expert_replay,is_GAIL, on_policy, work_dir, total_steps=1000000, warm_steps=10000,
             n_steps=1, n_updates=1,m_steps=1,discrim_steps = 1, n_checkpoint=None, n_eval=None, init_replay_buffers=None,
             init_replay_with_split=None, expert_replay_split_cfg = None,eval_cfg=None, replicate_init_buffer=1, split_expert_buffer=False ,num_trajs_per_demo_file=-1):
    logger = get_logger(env_cfg.env_name)

    import torch
    from mani_skill_learn.utils.torch import get_cuda_info
    replay_env.reset()
    


    if env_cfg.env_name.find('Drawer')>=0:    #! for drawer
        print('drawer')
        env_ids=['1000','1004','1005','1013','1016','1021','1024','1027','1032','1033','1035','1038','1040','1044',\
        '1045','1052','1054','1056','1061','1063','1066','1067','1076','1079','1082']
    elif env_cfg.env_name.find('Door')>=0:     #! for door
        print('door')
        env_ids=['1000','1001','1002','1006','1007','1014','1017','1018','1025','1026','1027','1028','1030','1031',\
        '1034','1036','1038','1039','1041','1042','1044','1045','1046','1047','1049','1051','1052','1054','1057',\
        '1060','1061','1062','1063','1064','1065','1067','1068','1073','1075','1077','1078','1081']
    else: 
        assert "you shouldn't get here"

    split_expert_buffer=True
    for env_id in env_ids:
        expert_replay[env_id]=build_replay(expert_replay_split_cfg)
        expert_replay[env_id].reset()

    trajs_split = []
    for _ in range(n_steps):
        trajs_split.append([])

    if (replay_model!=None):
        is_mbrl=1
        replay_model.reset()
        print("we are in mbrl")
    else:
        is_mbrl=0
        print("we are in mfrl")
    if init_replay_buffers is not None and init_replay_buffers != '':
        replay_env.restore(init_replay_buffers,
                           replicate_init_buffer, num_trajs_per_demo_file)
        logger.info(f'Initialize buffer with {len(replay_env)} samples')

    if init_replay_with_split is not None:
        assert is_seq_of(init_replay_with_split) and len(
            init_replay_with_split) == 2
        # For mani skill only
        from mani_skill.utils.misc import get_model_ids_from_yaml
        folder_root = init_replay_with_split[0]
        model_split_file = get_model_ids_from_yaml(init_replay_with_split[1])
        if init_replay_with_split[1] is None:
            files = [str(_) for _ in Path(folder_root).glob('*.h5')]
        else:
            files = [str(_) for _ in Path(folder_root).glob('*.h5')
                     if re.split('[_-]', _.name)[1] in model_split_file]
        replay_env.restore(files, replicate_init_buffer,
                           num_trajs_per_demo_file)

    tf_logs = ReplayMemory(total_steps)
    tf_logs.reset()
    tf_logger = TensorboardLogger(work_dir)

    checkpoint_dir = osp.join(work_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(agent)
    if rollout is not None:
        logger.info(
            f'Rollout state dim: {get_shape(rollout.recent_obs)}, action dim: {len(rollout.random_action())}')
        rollout.reset()
        episode_statistics = EpisodicStatistics(rollout.n)
        episode_statistics2 = EpisodicStatistics(rollout.n)
        total_episodes = 0
    else:
        # Batch RL
        if 'obs' not in replay_env.memory:
            logger.error('Empty replay buffer for Batch RL!')
        logger.info(
            f'State dim: {get_shape(replay_env["obs"])}, action dim: {replay_env["actions"].shape[-1]}')

    check_eval = EveryNSteps(n_eval)
    check_checkpoint = EveryNSteps(n_checkpoint)
    check_tf_log = EveryNSteps(1000)

    if warm_steps > 0:
        assert not on_policy
        assert rollout is not None
        trajectories = rollout.forward_with_policy(agent.policy, warm_steps)[0]
        #! trajectories are dict of keys {obs,actions,next_obs,rewards,dones,episode_dones}
        episode_statistics.push(
            trajectories['rewards'], trajectories['episode_dones'])
        replay_env.push_batch(**trajectories)
        vial=0
        if vial:
            with torch.no_grad():
                trajectories = to_torch(
                    trajectories, dtype='float32', device=torch.cuda.current_device(), non_blocking=True)
                pred = agent.model(
                    trajectories['obs'], trajectories['actions'])
                pred=to_np(pred)
                trajectories=to_np(trajectories)

            print(trajectories['obs']['pointcloud']['xyz'].shape)
            np.savetxt('visualization/seg0.txt',trajectories['obs']['pointcloud']['seg'][100:200,:,0])
            np.savetxt('visualization/seg1.txt',trajectories['obs']['pointcloud']['seg'][100:200,:,1])
            np.savetxt('visualization/seg2.txt',trajectories['obs']['pointcloud']['seg'][100:200,:,2])
            np.savetxt('visualization/obs0.txt',trajectories['obs']['pointcloud']['xyz'][100:200,:,0])
            np.savetxt('visualization/obs1.txt',trajectories['obs']['pointcloud']['xyz'][100:200,:,1])
            np.savetxt('visualization/obs2.txt',trajectories['obs']['pointcloud']['xyz'][100:200,:,2])

            # np.savetxt('visualization/nextobs0.txt',trajectories['next_obs']['pointcloud']['xyz'][100:200,:,0])
            # np.savetxt('visualization/nextobs1.txt',trajectories['next_obs']['pointcloud']['xyz'][100:200,:,1])
            # np.savetxt('visualization/nextobs2.txt',trajectories['next_obs']['pointcloud']['xyz'][100:200,:,2])
            # np.savetxt('visualization/nextseg0.txt',trajectories['next_obs']['pointcloud']['seg'][100:200,:,0])
            # np.savetxt('visualization/nextseg1.txt',trajectories['next_obs']['pointcloud']['seg'][100:200,:,1])
            # np.savetxt('visualization/nextseg2.txt',trajectories['next_obs']['pointcloud']['seg'][100:200,:,2])

            np.savetxt('visualization/next_obs_computed0.txt',trajectories['next_obs_computed']['pointcloud']['xyz'][100:200,:,0])
            np.savetxt('visualization/next_obs_computed1.txt',trajectories['next_obs_computed']['pointcloud']['xyz'][100:200,:,1])
            np.savetxt('visualization/next_obs_computed2.txt',trajectories['next_obs_computed']['pointcloud']['xyz'][100:200,:,2])
            np.savetxt('visualization/computed_nextseg0.txt',trajectories['next_obs_computed']['pointcloud']['seg'][100:200,:,0])
            np.savetxt('visualization/computed_nextseg1.txt',trajectories['next_obs_computed']['pointcloud']['seg'][100:200,:,1])
            np.savetxt('visualization/computed_nextseg2.txt',trajectories['next_obs_computed']['pointcloud']['seg'][100:200,:,2])

            np.savetxt('visualization/predobs0.txt',pred['pointcloud']['xyz'][100:200,:,0])
            np.savetxt('visualization/predobs1.txt',pred['pointcloud']['xyz'][100:200,:,1])
            np.savetxt('visualization/predobs2.txt',pred['pointcloud']['xyz'][100:200,:,2])
            np.savetxt('visualization/rew.txt',trajectories['rewards'][:100])
            print(trajectories['rewards'][100:200])
            print(np.max(trajectories['rewards'][100:200]))
            # exit()


        # chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        # chamfer_total1=0
        # points=500
        # cham=[]
        # for i in range(points):
        #     traj = rollout.forward_with_policy(agent.policy,8)[0]
        #     # traj = rollout.forward_with_policy(pol,8)[0]
        #     obs1=torch.from_numpy(traj['next_obs_computed']['pointcloud']['xyz']).cuda()
        #     obs2=torch.from_numpy(traj['obs']['pointcloud']['xyz']).cuda()
        #     dist1, dist2, idx1, idx2 = chamLoss(obs1,obs2)
        #     loss3 = (torch.mean(dist1)) + (torch.mean(dist2))
        #     chamfer_total1+=loss3.item()
        #     print(loss3,traj['rewards'])

        # print(chamfer_total1/points)
        # exit()
        if is_mbrl:
            replay_model.push_batch(**trajectories)
        rollout.reset()
        episode_statistics.reset_current()
        check_eval.check(warm_steps)
        check_checkpoint.check(warm_steps)
        check_tf_log.check(warm_steps)
        logger.info(f"Finish {warm_steps} warm-up steps!")

    steps = warm_steps
    total_updates = 0
    begin_time = datetime.now()
    max_ETA_len = None

    #! debug??????
    tmp_flag=0


    for iteration_id in itertools.count(1):
        tf_logs.reset()
        if rollout:
            episode_statistics.reset_history()
            episode_statistics2.reset_history()

        if on_policy:
            replay_env.reset()

        train_dict = {}
        print_dict = OrderedDict()

        update_time = 0
        time_begin_episode = time.time()


        if n_steps > 0:
            # For online RL
            collect_sample_time = 0
            cnt_episodes = 0
            num_done = 0

            if(is_mbrl):
                loss=agent.train_model(replay_env)
                print_dict["iteration id"] = iteration_id
                for i in range(len(loss)):
                    if(i==len(loss)-1):
                        print(f"iter {iteration_id} total loss is {float(loss[i])}")
                        print_dict[f'total loss']=float(loss[i])
                        continue

                    print(f"iter {iteration_id} loss {i+1} is {float(loss[i])}")
                    print_dict[f'pred loss{i+1}']=float(loss[i])



            """
            For on-policy algorithm, we will print training infos for every gradient batch.
            For off-policy algorithm, we will print training infos for every n_steps epochs.
            """
            print("now we are going to add environment steps")
            print(f"env buffer size is {len(replay_env)}")
            if (is_GAIL):
                print(f"tmp buffer size is {len(tmp_replay)}")
                tmp_replay.reset()
            if(is_mbrl):
                print(f"model buffer size is {len(replay_model)}")


            for env_id in env_ids:
                if(len(expert_replay[env_id])>0):print(f'env {env_id} have {len(expert_replay[env_id])} points')
            while num_done < n_steps and not (on_policy and num_done > 0):
                for _ in range (m_steps):
                    tmp_time = time.time()
                    recent_ids = rollout.recent_id()
                    trajectories, infos = rollout.forward_with_policy(agent.policy, n_steps, whole_episode=on_policy, merge= not(split_expert_buffer))

                    for k in range(n_steps):
                        selected_id = recent_ids[k]
                        # if(not save_points):
                        trajs_split[k].append(trajectories[k])
                        if trajectories[k]['dones']==1:
                            success_traj = concat_list_of_array(trajs_split[k])
                            expert_replay[str(selected_id)].push_batch(**success_traj)
                        if trajectories[k]['episode_dones']==1:
                            trajs_split[k]=[]
                        # else:
                        # if (trajectories[k]['rewards']>-7.5):
                        #     expert_replay[str(selected_id)].push_batch(**trajectories[k])

                    trajectories = concat_list_of_array(trajectories)
                    # trajectories['ids'] = np.array(trajectories['ids'])
                    infos = concat_list_of_array(infos)
                    #! ?????????reward?????????????????????buffer???
                    if (is_GAIL):
                        expert_rewards = agent.expert_reward(trajectories['obs'], trajectories['actions'])
                        episode_statistics2.push(expert_rewards, trajectories['episode_dones'])
                    # print(np.mean(expert_rewards))
                    episode_statistics.push(trajectories['rewards'], trajectories['episode_dones'])

                    collect_sample_time += time.time() - tmp_time

                    num_done += np.sum(trajectories['episode_dones'])
                    cnt_episodes += np.sum(trajectories['episode_dones'].astype(np.int32))
                    replay_env.push_batch(**trajectories)
                    if(tmp_replay is not None):
                        tmp_replay.push_batch(**trajectories)
                    steps += n_steps
                    if(is_mbrl):
                        # print(n_steps)
                        agent.model_rollout(replay_env,replay_model,n_steps,iter=iteration_id)

                for i in range(n_updates):
                    total_updates += 1
                    tmp_time = time.time()
                    if(is_mbrl):
                        tf_logs.push(
                            **agent.update_parameters(replay_model, memory2=replay_env,expert_replay=expert_replay, updates=total_updates,iter=iteration_id))
                    else:
                        tf_logs.push(
                            **agent.update_parameters(replay_env,expert_replay=expert_replay, updates=total_updates))
                    update_time += time.time() - tmp_time
            if(is_GAIL):
                tmploss=0
                exploss=0
                for _i_ in range(discrim_steps):
                    tmp_time = time.time()
                    el, tl = agent.update_discriminator(expert_replay, tmp_replay, expert_split = split_expert_buffer)
                    tmploss += tl
                    exploss += el
                tmploss /= discrim_steps
                exploss /= discrim_steps           
                print_dict['episode_length'], print_dict['expert_reward'] = episode_statistics2.get_mean()


            total_episodes += cnt_episodes
            train_dict['num_episode'] = int(cnt_episodes)
            train_dict['total_episode'] = int(total_episodes)
            train_dict['episode_time'] = time.time() - time_begin_episode
            train_dict['collect_sample_time'] = collect_sample_time

            print_dict['episode_length'], print_dict['episode_reward'] = episode_statistics.get_mean()

            if(is_GAIL):
                print_dict['fake_sample_loss'] = tmploss
                print_dict['expert_sample_loss'] = exploss
        else:
            # For offline RL
            tf_logs.reset()
            for i in range(n_updates):
                steps += 1
                total_updates += 1
                tmp_time = time.time()
                tf_logs.push(
                    **agent.update_parameters(replay_env, updates=total_updates))
                update_time += time.time() - tmp_time
        train_dict['update_time'] = update_time
        train_dict['total_updates'] = int(total_updates)
        train_dict['buffer_size'] = len(replay_env)
        train_dict['memory'] = get_total_memory('G', True)
        train_dict['cuda_mem'] = get_total_memory('G', True)

        train_dict.update(get_cuda_info(device=torch.cuda.current_device()))

        print_dict.update(tf_logs.tail_mean(n_updates))
        print_dict['memory'] = get_total_memory('G', False)
        print_dict.update(get_cuda_info(
            device=torch.cuda.current_device(), number_only=False))

        print_info = dict_to_str(print_dict)

        percentage = f'{(steps / total_steps) * 100:.0f}%'
        passed_time = td_format(datetime.now() - begin_time)
        ETA = td_format((datetime.now() - begin_time) *
                        (total_steps / (steps - warm_steps) - 1))
        if max_ETA_len is None:
            max_ETA_len = len(ETA)

        logger.info(
            f'{steps}/{total_steps}({percentage}) Passed time:{passed_time} ETA:{ETA} {print_info}')
        if check_tf_log.check(steps):
            train_dict.update(dict(print_dict))
            tf_logger.log(train_dict, n_iter=steps, eval=False)

        if check_checkpoint.check(steps):
            standardized_ckpt_step = check_checkpoint.standard(steps)
            model_path = osp.join(
                checkpoint_dir, f'model_{standardized_ckpt_step}.ckpt')
            buffer_path = osp.join(
                checkpoint_dir, 'buffer.h5')
            # replay_env.to_h5(buffer_path)
            logger.info(
                f'Save model at step: {steps}.The model will be saved at {model_path}')
            agent.to_normal()
            save_checkpoint(agent, model_path)

            agent.recover_data_parallel()
        if check_eval.check(steps):
            standardized_eval_step = check_eval.standard(steps)
            logger.info(f'Begin to evaluate at step: {steps}. '
                        f'The evaluation info will be saved at eval_{standardized_eval_step}')
            eval_dir = osp.join(work_dir, f'eval_{standardized_eval_step}')
            # eval_dir=None
            agent.eval()
            torch.cuda.empty_cache()
            lens, rewards, finishes, selected_ids, target_ids = evaluator.run(agent, **eval_cfg, work_dir=eval_dir)
            torch.cuda.empty_cache()
            save_eval_statistics(eval_dir, lens, rewards, finishes, selected_ids, target_ids, logger)
            agent.train()

            eval_dict = {}
            eval_dict['mean_length'] = np.mean(lens)
            eval_dict['std_length'] = np.std(lens)
            eval_dict['mean_reward'] = np.mean(rewards)
            eval_dict['std_reward'] = np.std(rewards)
            eval_dict['success_rate'] = np.mean(finishes)
            tf_logger.log(eval_dict, n_iter=steps, eval=True)

        if steps >= total_steps:
            break
    if n_checkpoint:
        print(f'Save checkpoint at final step {total_steps}')
        agent.to_normal()
        save_checkpoint(agent, osp.join(
            checkpoint_dir, f'model_{total_steps}.ckpt'))