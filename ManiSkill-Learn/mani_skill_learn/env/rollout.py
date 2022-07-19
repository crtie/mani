import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import numpy.matlib 
import torch
import cv2
import sys
import copy
from urdfpy import URDF
from mani_skill_learn.utils.data import (to_np, concat_list_of_array, stack_dict_of_list_array, flatten_dict,
                             compress_size, unsqueeze, stack_list_of_array)
from mani_skill_learn.utils.meta import dict_of
from .builder import ROLLOUTS
from .env_utils import build_env, true_done
from ..utils.oracle import door_orientation as door_ori
sys.path.append("/home/ruichentie/mani/ChamferDistancePytorch")
robot = URDF.load('/home/ruichentie/ManiSkill/mani_skill/assets/robot/sciurus/A2_single.urdf')
# import chamfer3D.dist_chamfer_3D
# chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
@ROLLOUTS.register_module()
class Rollout:
    def __init__(self, env_cfg, worker_id=None, use_cost=False, with_info=False, reward_only=True,compute_next_obs=False, **kwargs):
        self.n = 1
        self.worker_id = worker_id
        self.env = build_env(env_cfg)
        if hasattr(self.env, 'seed') and worker_id is not None:
            # Assume parallel-run is multi-process
            self.env.seed(np.random.randint(0, 10000) + worker_id)
        self.iscost = -1 if use_cost else 1
        self.reward_only = reward_only
        self.with_info = with_info
        self.recent_obs = None
        self.step = 0
        self.reset()
        self.compute_next_obs = compute_next_obs
        if self.compute_next_obs:
            print('we compute next_obs')
        else:
            print('we use next obs sampled from env')

    def reset(self, **kwargs):
        self.step = 0
        reset_kwargs = {}
        kwargs = deepcopy(dict(kwargs))
        if 'level' in kwargs:
            reset_kwargs['level'] = kwargs['level']
        # flush_print(self.worker_id, 'Begin reset')
        self.recent_obs = compress_size(self.env.reset(**reset_kwargs))
        # flush_print(self.worker_id, 'End reset')

    def random_action(self):
        return self.env.action_space.sample()

    def cur_id(self):
        return self.env.selected_id

    def forward_with_reset(self, states=None, actions=None):
        """
        :param states: [n, m] n different env states
        :param actions: [n, c, a] n sequences of actions
        :return: rewards [n, c]
        """
        # for CEM only
        assert self.reward_only
        rewards = []
        for s, a in zip(states, actions):
            self.env.set_state(s)
            reward_episode = []
            for action in a:
                ob, r, done, _ = self.env.step(action)
                reward_episode.append(r * self.iscost)
            rewards.append(reward_episode)
        rewards = np.array(rewards, dtype=np.float32)
        return rewards

    def forward_with_policy(self, pi=None, num=1, whole_episode=False):
        assert not self.reward_only and self.recent_obs is not None
        obs, next_obs, actions, rewards, dones, episode_dones = [], [], [], [], [], []
        infos = defaultdict(list)

        if pi is not None:
            import torch
            from mani_skill_learn.utils.data import to_torch
            device = pi.device

        for i in itertools.count(1):
            if pi is not None:
                with torch.no_grad():
                    recent_obs = to_torch(self.recent_obs, dtype='float32', device=device)
                    a = to_np(pi(unsqueeze(recent_obs, axis=0)))[0]
            else:
                a = self.random_action()
            ob, r, done, info = self.env.step(a)

            ob = compress_size(ob)
            self.step += 1
            episode_done = done
            # done = done if self.step < self.env._max_episode_steps else False
            done = true_done(done, info)

            obs.append(self.recent_obs)
            next_obs.append(ob)
            actions.append(a)
            rewards.append(compress_size(r * self.iscost))

            dones.append(done)
            episode_dones.append(episode_done)
            if self.with_info:
                info = flatten_dict(info)
                for key in info:
                    infos[key].append(info[key])
            self.recent_obs = ob
            if episode_done:
                self.reset()
            if i >= num and (episode_done or not whole_episode):
                break
        ret = dict_of(obs, actions, next_obs, rewards, dones, episode_dones)
        ret = stack_dict_of_list_array(ret)
        infos = stack_dict_of_list_array(dict(infos))
        return ret, infos
    def compute_agent_point_shift(self, pcd, seg ,agent_state_current,agent_state_next):
        # pcd: (N*3) :属于agent的点的xyz
        # seg: (N*20) 前两个不管，后面18这个点属于agent的不同link
        #agent_state: 前后两帧的agent state，是字典
        # {'fingers_pos' 'fingers_vel' 'qpos' 'qvel' 'base_pos' 'base_orientation' 'base_vel'  'base_ang_vel':}
        #! 一共13个能动的joint，base pos2，base ori1，qpos10
        ret = pcd.copy()
        num_points=pcd.shape[0]
        homo = np.ones(num_points)[:,None]
        ret=np.concatenate((ret,homo),axis=1)
        joint_names = ['root_x_axis_joint','root_y_axis_joint','root_z_rotation_joint','linear_actuator_height','right_panda_joint1',\
        'right_panda_joint2','right_panda_joint3','right_panda_joint4','right_panda_joint5','right_panda_joint6','right_panda_joint7',\
        'right_panda_finger_joint1','right_panda_finger_joint2']
        active_links=['right_panda_link1','right_panda_link2','right_panda_link3','right_panda_link4',\
        'right_panda_link5','right_panda_link6','right_panda_link7','right_panda_link8',\
        'right_panda_hand','right_panda_leftfinger','right_panda_rightfinger']
        state_dict1={}
        state_dict1 = {joint_names[i+3]:agent_state_current['qpos'][i] for i in range(10)}
        state_dict1['root_x_axis_joint'] = agent_state_current['base_pos'][0]
        state_dict1['root_y_axis_joint'] = agent_state_current['base_pos'][1]
        state_dict1['root_z_rotation_joint'] = agent_state_current['base_orientation']
        fk1 = URDF.link_fk(robot,state_dict1,links=active_links)

        state_dict2 = {joint_names[i+3]:agent_state_next['qpos'][i] for i in range(10)}
        state_dict2['root_x_axis_joint'] = agent_state_next['base_pos'][0]
        state_dict2['root_y_axis_joint'] = agent_state_next['base_pos'][1]
        state_dict2['root_z_rotation_joint'] = agent_state_next['base_orientation']
        state_dict2['linear_actuator_height'] = agent_state_next['qpos'][0]
        fk2 = URDF.link_fk(robot,state_dict2,links=active_links)
        # print(fk1)
        # print(robot.links,robot.link_map)

        for i in range(9):
            mask = np.where(seg[:,i+9]==True)[0] #! 属于当前link的点
            # print(len(mask))
            T_current = np.array(fk1[robot.link_map[active_links[i]]])
            T_next = np.array(fk2[robot.link_map[active_links[i]]])
            # print(T_next.shape)
            if len(mask)==0:
                continue

            # print(T_current)
            ret[mask] = np.dot(T_next,np.dot(np.linalg.inv(T_current),ret[mask].T)).T
        left_finger_mask = np.where(seg[:,18]==True)[0]
        right_finger_mask = np.where(seg[:,19]==True)[0]
        
        # print(ret.shape)
        #! 把齐次坐标变回正常坐标
        ret = ret /ret[:,3][:,None]
        ret=ret[:,:3]
        ret[left_finger_mask] += (agent_state_next['fingers_pos'] - agent_state_current['fingers_pos'])[:3]
        ret[right_finger_mask] += (agent_state_next['fingers_pos'] - agent_state_current['fingers_pos'])[3:]
        return ret



    def forward_single(self, action=None):
        """
        :param action: [a] one action
        :return: all information
        """
        # print('I\' here')
        assert not self.reward_only and self.recent_obs is not None
        if action is None:
            action = self.random_action()
        actions = action
        # actions=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
        if self.compute_next_obs:
            obs = self.env.get_obs(detailed_mask_agent=True)
            agent_state_current = self.env.agent.get_state(by_dict=True, with_controller_state=False)
            num_points = obs['pointcloud']['xyz'].shape[0]
            qpos_current = self.env.cabinet.get_qpos()[self.env.target_index_in_active_joints]
            next_obs, rewards, dones, info = self.env.step(actions)
            agent_state_next = self.env.agent.get_state(by_dict=True, with_controller_state=False)
            qpos_next = self.env.cabinet.get_qpos()[self.env.target_index_in_active_joints]
            delta_pos = qpos_next - qpos_current
            mask_all = np.arange(num_points)
            mask_cabinet = np.where(~obs['pointcloud']['seg'][:,2:].any(-1, keepdims=True))[0]
            mask_agent =  np.where(obs['pointcloud']['seg'][:,2:].any(-1, keepdims=True))[0]
            mask_target = np.where(obs['pointcloud']['seg'][:,1]==True)[0]
            # next_obs_computed = copy.deepcopy(obs)
            
            next_obs_computed = next_obs
            next_obs_computed['pointcloud']['xyz']= obs['pointcloud']['xyz'].copy()
            next_obs_computed['pointcloud']['rgb']= obs['pointcloud']['rgb'].copy()
            next_obs_computed['pointcloud']['seg']= obs['pointcloud']['seg'].copy()

            next_obs_computed['pointcloud']['xyz'][mask_agent] = self.compute_agent_point_shift(obs['pointcloud']['xyz'][mask_agent],
            obs['pointcloud']['seg'][mask_agent] ,agent_state_current,agent_state_next)
            # next_obs_computed['pointcloud']['xyz'][mask_target] =0
            # next_obs_computed['pointcloud']['xyz'][mask_target][:,0] =0
            if (self.env.traget_link_type == 'prismatic'): #! 抽屉
                # print("拉抽屉")
                to_add = np.array([-delta_pos,0,0])
                next_obs_computed['pointcloud']['xyz'][mask_target]+=to_add
            else: 
                print('you should not get here')
                exit()
            if (self.env.traget_link_type == 'revolute'): #! 门
                # print("开门")
                env_id = str(self.cur_id())
                ori_dict = door_ori.door_orientation
                assert env_id in ori_dict.keys(), "env id isn't valid"
                joint_id = self.env.target_index
                target_door_ori = ori_dict[env_id][int(joint_id)]
                assert target_door_ori in ['l','r'], "wrong door orientation"
                if target_door_ori == 'l':
                    pivot = [np.min(obs['pointcloud']['xyz'][:,0]),np.min(obs['pointcloud']['xyz'][:,1]),0]
                elif target_door_ori == 'r':
                    pivot = [np.max(obs['pointcloud']['xyz'][:,0]),np.min(obs['pointcloud']['xyz'][:,1]),0]
                aligned_coord = obs['pointcloud']['xyz'][mask_target] - pivot #! 这就对齐到以转轴为z轴的坐标系去了
                R_vec = np.array([0,0,1]) * delta_pos
                R= cv2.Rodrigues(R_vec)[0]
                rotated_coord = (np.matmul(R,aligned_coord.T)).T + pivot
                next_obs_computed['pointcloud']['xyz'][mask_target] = rotated_coord
            seg_agent = obs['pointcloud']['seg'][:,2:].any(-1, keepdims=True).squeeze()

            obs['pointcloud']['seg'] = obs['pointcloud']['seg'][:,:3]
            obs['pointcloud']['seg'][:,2] = seg_agent 
            next_obs_computed['pointcloud']['seg'] = next_obs_computed['pointcloud']['seg'][:,:3]
            next_obs_computed['pointcloud']['seg'][:,2] = seg_agent 
            rewards *= self.iscost
            episode_dones = dones

            next_obs_computed = compress_size(next_obs_computed)
            rewards = compress_size(rewards)

            self.step += 1
            dones = true_done(dones, info)
            ret = dict_of(obs, actions, next_obs, next_obs_computed, rewards, dones, episode_dones)
            self.recent_obs = next_obs
            # print(self.recent_obs['pointcloud']['seg'].shape)
            if self.with_info:
                info = flatten_dict(info)
            else:
                info = {}
            if episode_dones:
                self.reset()
            return ret, info

        else: 
            obs = self.recent_obs
            next_obs, rewards, dones, info = self.env.step(actions)
            rewards *= self.iscost
            episode_dones = dones

            next_obs = compress_size(next_obs)
            rewards = compress_size(rewards)

            self.step += 1
            dones = true_done(dones, info)
            ret = dict_of(obs, actions, next_obs, rewards, dones, episode_dones)
            self.recent_obs = next_obs
            # print(self.recent_obs['pointcloud']['seg'].shape)
            if self.with_info:
                info = flatten_dict(info)
            else:
                info = {}
            if episode_dones:
                self.reset()
            return ret, info

    def close(self):
        if self.env:
            del self.env


@ROLLOUTS.register_module()
class BatchRollout:
    def __init__(self, env_cfg, num_procs=20, synchronize=True, reward_only=False,compute_next_obs=False ,**kwargs):
        self.n = num_procs
        self.synchronize = synchronize
        self.reward_only = reward_only
        self.workers = []
        self.compute_next_obs = compute_next_obs

        if synchronize:
            from ..env.parallel_runner import NormalWorker as Worker
        else:
            from ..env.torch_parallel_runner import TorchWorker as Worker
            print("This will consume a lot of memory due to cuda")
        for i in range(self.n):
            self.workers.append(Worker(Rollout, i, env_cfg, reward_only=reward_only,compute_next_obs=compute_next_obs, **kwargs))

    def reset(self, **kwargs):
        for i in range(self.n):
            self.workers[i].call('reset', **kwargs)
        for i in range(self.n):
            self.workers[i].get()

    @property
    def recent_obs(self):
        for i in range(self.n):
            self.workers[i].get_attr('recent_obs')
        return stack_list_of_array([self.workers[i].get() for i in range(self.n)])

    def recent_id(self):
        for i in range(self.n):
            self.workers[i].call('cur_id')
        return np.array([self.workers[i].get() for i in range(self.n)])

    def random_action(self):
        for i in range(self.n):
            self.workers[i].call('random_action')
        return np.array([self.workers[i].get() for i in range(self.n)])

    def forward_with_reset(self, states=None, actions=None):
        from .parallel_runner import split_list_of_parameters
        paras = split_list_of_parameters(self.n, states=states, actions=actions)
        n = len(paras)
        for i in range(n):
            args_i, kwargs_i = paras[i]
            self.workers[i].call('forward_with_reset', *args_i, **kwargs_i)
        reward = [self.workers[i].get() for i in range(n)]
        reward = concat_list_of_array(reward)
        return reward

    def forward_with_policy(self, policy, num, whole_episode=False, merge=True):
        from mani_skill_learn.utils.math import split_num
        n, running_steps = split_num(num, self.n)
        batch_size = max(running_steps)
        if self.synchronize and policy is not None:
            """
            When the we run with random actions, it is ok to use asynchronizedly
            """
            device = policy.device
            trajectories = defaultdict(lambda: [[] for i in range(n)])
            infos = defaultdict(lambda: [[] for i in range(n)])
            for i in range(batch_size):
                current_n = 0
                for j in range(n):
                    if i < running_steps[j]:
                        current_n += 1
                assert current_n > 0
                if policy is None:
                    action = None
                else:
                    import torch
                    from mani_skill_learn.utils.data import to_torch
                    with torch.no_grad():
                        recent_obs = to_torch(self.recent_obs, dtype='float32', device=device)
                        action = to_np(policy(recent_obs))[:current_n]

                for j in range(current_n):
                    self.workers[j].call('forward_single', action=None if action is None else action[j])

                for j in range(current_n):
                    traj, info = self.workers[j].get()
                    for key in traj:
                        trajectories[key][j].append(traj[key])
                    for key in info:
                        infos[key][j].append(info[key])
            trajectories = [{key: stack_list_of_array(trajectories[key][i]) for key in trajectories} for i in range(n)]
            infos = [{key: stack_list_of_array(infos[key][i]) for key in infos} for i in range(n)]
        else:
            print('you can\'t go here')
        #     for i in range(n):
        #         if policy is not None:
        #             assert not self.synchronize
        #         self.workers[i].call('forward_with_policy', pi=policy, num=running_steps[i],
        #                              whole_episode=whole_episode)
        #     ret = [self.workers[i].get() for i in range(n)]
        #     trajectories = [ret[i][0] for i in range(n)]
        #     infos = [ret[i][1] for i in range(n)]

        if merge:
            trajectories = concat_list_of_array(trajectories)
            infos = concat_list_of_array(infos)
            """
            Concat: [Process 0, Process1, ..., Process n]
            """
        return trajectories, infos

    def close(self):
        for worker in self.workers:
            worker.call('close')
            worker.close()
