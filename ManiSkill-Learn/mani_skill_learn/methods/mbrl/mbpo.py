"""
! Model Based Reinforcement Learning
"""
import sys
sys.path.append("/home/ruichentie/mani/ChamferDistancePytorch")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy
from mani_skill_learn.networks import build_model, hard_update, soft_update
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch,to_np,merge_dict, dict_to_seq,concat_list_of_array,state_dict2tensor,tensor2state_dict,unsqueeze
from ..builder import MBRL
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.utils.meta import get_logger, get_total_memory, td_format
from torch.utils.tensorboard import SummaryWriter
import chamfer3D.dist_chamfer_3D
from mani_skill_learn.env.torch_parallel_runner import TorchWorker as Worker
import time
writer = SummaryWriter("logs")


@MBRL.register_module()
class MBPO(BaseAgent):
    def __init__(self, policy_cfg, value_cfg, model_cfg, obs_shape, action_shape, action_space, batch_size=128, gamma=0.99,
                 update_coeff=0.005, alpha=0.2, target_update_interval=1, max_iter_use_real_data=1000,use_expert=0,model_updates=256,
                 data_generated_by_model=4 ,max_model_horizon=1, automatic_alpha_tuning=True, cem=False, cem_cfg=None,
                 alpha_optim_cfg=None):
        super(MBPO, self).__init__()
        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        value_optim_cfg = value_cfg.pop("optim_cfg")
        model_optim_cfg = model_cfg.pop("optim_cfg")

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning
        self.max_iter_use_real_data = max_iter_use_real_data
        self.use_expert = use_expert
        self.model_updates = model_updates
        self.data_generated_by_model = data_generated_by_model
        self.max_model_horizon = max_model_horizon
        self.use_cem = cem
        self.cem_cfg = cem_cfg
        self.ac_dim =action_shape

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space
        value_cfg['obs_shape'] = obs_shape
        value_cfg['action_shape'] = action_shape

        model_cfg['obs_shape'] = obs_shape
        model_cfg['action_shape'] = action_shape

        self.policy = build_model(policy_cfg)
        self.critic = build_model(value_cfg)
        self.model = build_model(model_cfg)
        

        #print("now we are printing model")
        # print(self.model)

        self.target_critic = build_model(value_cfg)
        hard_update(self.target_critic, self.critic)
        #! 这一步就是把这俩弄成一样的

        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -np.prod(action_shape)
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)
        self.model_optim = build_optimizer(self.model, model_optim_cfg)

    def train_model(self, replay_env):
        # print(self.model)
        chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        for i in range(self.model_updates):
            sampled_batch = replay_env.sample(self.batch_size)
            sampled_batch = to_torch(
                sampled_batch, dtype='float32', device=self.device, non_blocking=True)
            for key in sampled_batch:
                if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                    sampled_batch[key] = sampled_batch[key][..., None]
            pred = self.model(
                  sampled_batch['obs'],sampled_batch['actions'])
            if isinstance (pred ,dict):#! 输入是点云
                loss1=F.mse_loss(sampled_batch['rewards'].squeeze(),pred['rewards'])
                # print(sampled_batch['rewards'].shape,pred['reward'].shape)
                loss2=F.mse_loss(sampled_batch['next_obs']['state'],pred['state'])
                # print(sampled_batch['next_obs']['state'].shape,pred['state'].shape)
                # print(sampled_batch['next_obs']['pointcloud']['xyz'].shape,pred['pointcloud']['xyz'].shape)
                dist1, dist2, idx1, idx2 = chamLoss(sampled_batch['next_obs']['pointcloud']['xyz'],pred['pointcloud']['xyz'])
                loss3 = (torch.mean(dist1)) + (torch.mean(dist2))
                # print(loss1,loss2,loss3)
                # print(loss1,loss2,loss3)
                mse_loss=loss1+loss2+100*loss3
                # print(loss1.item(),loss2.item(),loss3.item())
                # print(mse_loss)
                # print(f'iter {i} ,total loss is {mse_loss}')
            else:  #! 输入是state
                labels=torch.cat([sampled_batch['rewards'] , sampled_batch['next_obs']-sampled_batch['obs']],dim=-1)
                mse_loss = F.mse_loss(pred,labels)

            self.model_optim.zero_grad()
            mse_loss.backward()
            self.model_optim.step()
        return loss1,loss2,loss3,mse_loss

    def model_rollout(self, replay_env, replay_model,n_steps,iter):
        #! 每在真实环境走n_steps步调用一次本函数，故一共要造出self.data_generated_by_model * n_steps个点，把这些点平均分给current_model_horizon
        total_generated_data = self.data_generated_by_model * n_steps
        # if iter<=50:
        #     current_horizon = 1
        # elif iter>=self.max_iter_use_real_data:
        #     current_horizon = self.max_model_horizon
        # else:
        #     a = (self.max_model_horizon-1)/(self.max_iter_use_real_data-50)
        #     b = 1-50*a
        #     current_horizon = int(a*iter+b)
        # sample_num = int(total_generated_data/current_horizon)
        sampled_batch = replay_env.sample(total_generated_data)
        # for h in range(current_horizon):
        sampled_batch = to_torch(
            sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        # print(sampled_batch)
        with torch.no_grad():
            next_action = self.policy(
                sampled_batch['next_obs'], mode='sample')
            pred = self.model(
                sampled_batch['next_obs'], next_action)
            if isinstance(pred,dict): #! take pointcloud as input
                pred_obs={}
                pred_obs['pointcloud']=pred['pointcloud']
                pred_obs['state']=pred['state']
                pred_reward=pred['rewards']
        rollout = dict()
        #! trajectories are dict of keys {obs,actions,next_obs,rewards,dones,episode_dones}
        rollout['obs'] = to_np(sampled_batch['next_obs'])
        rollout['actions'] = to_np(next_action)
        rollout['next_obs'] = to_np(pred_obs)
        rollout['rewards'] = to_np(pred_reward)
        rollout['dones'] = np.zeros_like(pred_reward.cpu())
        rollout['episode_dones'] = np.zeros_like(pred_reward.cpu())
        replay_model.push_batch(**rollout)
        # sampled_batch = copy.deepcopy(rollout)

    def update_parameters(self, memory1,updates,memory2=None,expert_replay=None,alpha=0.05,iter=0):
        num_expert_replay_is_not_null=0
        if expert_replay is not None:
            for env_id in expert_replay.keys():
                if(len(expert_replay[env_id])>0):
                    num_expert_replay_is_not_null+=1
        beta_each = 0.01 if num_expert_replay_is_not_null>0 else 0
        if(iter==0): #! static rate
            sampled_batch1 = memory1.sample(int(self.batch_size*alpha))
            sampled_batch2 = memory2.sample(self.batch_size-int(self.batch_size*alpha))
        else:
            if self.use_expert:
                alpha=min(0.8,float(iter/self.max_iter_use_real_data))
                sampled_batch3=[]
                expert_data = 0
                for env_id in expert_replay.keys():
                    if(len(expert_replay[env_id])>0):
                        data_to_sample=min(len(expert_replay[env_id]),int(self.batch_size*beta_each))
                        expert_data += data_to_sample
                        # print(expert_data)
                        sampled_batch3.append(expert_replay[env_id].sample(data_to_sample))
                rest =self.batch_size - expert_data
                data_from_model = int(alpha*rest)
                data_from_env = rest-data_from_model

                sampled_batch1 = memory1.sample(data_from_model)
                sampled_batch2 = memory2.sample(data_from_env)

                permutation = list(np.random.permutation(rest))
                for key in sampled_batch1:
                    if not isinstance(sampled_batch1[key], dict) and sampled_batch1[key].ndim == 1:
                        sampled_batch1[key] = sampled_batch1[key][..., None]

                for key in sampled_batch2:
                    if not isinstance(sampled_batch2[key], dict) and sampled_batch2[key].ndim == 1:
                        sampled_batch2[key] = sampled_batch2[key][..., None]
                
                sampled_batch = merge_dict(sampled_batch1,sampled_batch2,permutation)

                for i in range(len(sampled_batch3)):
                    for key in sampled_batch3[i]:
                        if not isinstance(sampled_batch3[i][key], dict) and sampled_batch3[i][key].ndim == 1:
                            sampled_batch3[i][key] = sampled_batch3[i][key][..., None]
                    sampled_batch=merge_dict(sampled_batch,sampled_batch3[i])

            else:
                alpha=min(0.8,float(iter/self.max_iter_use_real_data))
                alpha=0
                #print(f'{alpha*100}percentage data are collected from model buffer')
                # sampled_batch1 = memory1.sample(max(1,int(self.batch_size*alpha)))
                sampled_batch2 = memory2.sample(self.batch_size)
                sampled_batch={}
                # for key in sampled_batch1:
                #     if not isinstance(sampled_batch1[key], dict) and sampled_batch1[key].ndim == 1:
                #         sampled_batch1[key] = sampled_batch1[key][..., None]
                for key in sampled_batch2:
                    if not isinstance(sampled_batch2[key], dict) and sampled_batch2[key].ndim == 1:
                        sampled_batch2[key] = sampled_batch2[key][..., None]
                # permutation = list(np.random.permutation(self.batch_size))
        # for key in sampled_batch:
        #     if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
        #         sampled_batch[key] = sampled_batch[key][..., None]
                # sampled_batch=merge_dict(sampled_batch1,sampled_batch2,permutation)
                sampled_batch = sampled_batch2


        sampled_batch = to_torch(
            sampled_batch, dtype='float32', device=self.device, non_blocking=True)

        with torch.no_grad():  # ! 这里为啥不用梯度
            next_action, next_log_prob = self.policy(sampled_batch['next_obs'], mode='all')[:2]
            q_next_target = self.target_critic(sampled_batch['next_obs'], next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * min_q_next_target
        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        abs_critic_error = torch.abs(q - q_target.repeat(1, q.shape[-1]))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        pi, log_pi = self.policy(sampled_batch['obs'], mode='all')[:2]
        q_pi = self.critic(sampled_batch['obs'], pi)
        q_pi_min = torch.min(q_pi, dim=-1, keepdim=True).values
        policy_loss = -(q_pi_min - self.alpha * log_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_alpha_tuning:
            alpha_loss = self.log_alpha.exp() * (-(log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)

        return {
            'critic_loss': critic_loss.item(),
            'max_critic_abs_err': abs_critic_error.max().item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'log_pi': torch.mean(log_pi).item(),
            'num_expert': num_expert_replay_is_not_null,
        }
    def cem_rollout(self,obs,mode):
        #! obs: 8个obs，每个有pointcloud和state
        obs = to_torch(obs,dtype='float32',device=self.device)
        origin_param = self.policy.state_dict()
        params = state_dict2tensor(origin_param) #! tensor of shape (282676)
        num_params = params.shape[0]
        ret=np.zeros((self.ac_dim))
        start_time=time.time()
        #! 还是试试对一个sequence里的所有点都用一样的weight noise吧
        weight_noise = np.random.normal(0,1,(self.cem_cfg['cem_num_sequances'],num_params))
        # print(weight_noise.shape)
        mean = np.mean(weight_noise,axis=0,keepdims=False)
        std = np.std(weight_noise,axis=0,keepdims=False) #! params
        # print(mean.shape)
        for j in range(self.cem_cfg['cem_iterations']):
            start_iter_time=time.time()
            rewards = to_np(self.evaluate_candidate_sequences(weight_noise,obs))
            elite_idx = np.argsort(rewards)[-self.cem_cfg['cem_num_elites']:]
            elite_sequences = weight_noise[elite_idx]
            # print(mean.shape)
            mean *= 1 - self.cem_cfg['cem_alpha'] 
            mean += self.cem_cfg['cem_alpha'] * np.mean(elite_sequences, axis=0, keepdims=False) 
            if j == self.cem_cfg['cem_iterations']:
                break
            std *= 1 - self.cem_cfg['cem_alpha']  
            std *= self.cem_cfg['cem_alpha'] * np.std(elite_sequences, axis=0, keepdims=False)
            weight_noise = np.random.normal(mean, std,(self.cem_cfg['cem_num_sequances'],num_params))
            end_iter_time=time.time()
            # print(f'we spend {end_iter_time-start_iter_time} seconds for cem iteration')
        params_tmp = params + to_torch(mean).cuda()
        state_dict_tmp = tensor2state_dict(params_tmp,origin_param)
        self.policy.load_state_dict(state_dict_tmp)
        ret =to_np(self.policy(obs,mode='eval'))
        end_time = time.time()
        print(f'we spend {end_time-start_time} seconds for one step')
        self.policy.load_state_dict(origin_param)
        return np.squeeze(ret)



    def evaluate_candidate_sequences(self,weight_noise,obs):
        weight_noise=to_torch(weight_noise).cuda()
        #! obs:单帧obs    weight_noise: num_sequences* num_params
        origin_param = self.policy.state_dict()
        params = state_dict2tensor(origin_param).cuda() #! tensor of shape (282676)
        num_params = params.shape[0]
        rewards = torch.zeros((weight_noise.shape[0])).cuda()
        # print(rewards.shape)
        for i in range(weight_noise.shape[0]):#!每一条sequence
            # obs_tmp = copy.deepcopy(obs)
            # params_tmp = copy.deepcopy(params)
            # params_tmp = params + weight_noise[i]
            # state_dict_tmp = tensor2state_dict(params_tmp,origin_param)
            # self.policy.load_state_dict(state_dict_tmp)
            for j in range(self.cem_cfg['cem_horizon']):#!每一步
                action = self.policy(obs_tmp,mode='eval')
                pred = self.model(obs_tmp,action)
                obs_tmp['pointcloud'] = pred['pointcloud']
                obs_tmp['state'] = pred['state']
                rewards[i]+= pred['rewards'].item()*(self.gamma**j)
                if(j==weight_noise.shape[1]-1):
                    pi=self.policy(obs_tmp,mode='eval')
                    q_pi = self.critic(obs_tmp, pi)
                    q_pi_min = torch.min(q_pi, dim=-1, keepdim=False).values[0].item()
                    rewards[i]+=q_pi_min*(self.gamma**weight_noise.shape[1])
        self.policy.load_state_dict(origin_param)
        return rewards






