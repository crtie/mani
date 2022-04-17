"""
! Model Based Reinforcement Learning
"""
import sys
sys.path.append("/home/ruichentie/mani/ChamferDistancePytorch")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mani_skill_learn.networks import build_model, hard_update, soft_update
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch,to_np,merge_dict
from ..builder import MBRL
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.utils.meta import get_logger, get_total_memory, td_format
from torch.utils.tensorboard import SummaryWriter
import chamfer3D.dist_chamfer_3D
writer = SummaryWriter("logs")


@MBRL.register_module()
class MBPO(BaseAgent):
    def __init__(self, policy_cfg, value_cfg, model_cfg, obs_shape, action_shape, action_space, batch_size=128, gamma=0.99,
                 update_coeff=0.005, alpha=0.2, target_update_interval=1, max_iter_use_real_data=1000, automatic_alpha_tuning=True,
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
        self.max_iter_use_real_data=max_iter_use_real_data

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
        for i in range(256):
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
                # print(mse_loss)
            else:  #! 输入是state
                labels=torch.cat([sampled_batch['rewards'] , sampled_batch['next_obs']-sampled_batch['obs']],dim=-1)
                mse_loss = F.mse_loss(pred,labels)

            self.model_optim.zero_grad()
            mse_loss.backward()
            self.model_optim.step()
        return loss1,loss2,loss3

    def model_rollout(self, replay_env, replay_model):
          # ! 每次造4倍于环境步的model数据
        sampled_batch = replay_env.sample(32)
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


            else:#! state input
                pred[:, :, 1:] += sampled_batch['next_obs']
                num_models, batch_size, _ = ensemble_model_means.shape
                model_idxes = torch.randint(num_models, (batch_size,))
                batch_idxes = torch.arange(0, batch_size)
                #print(model_idxes)
                samples = pred[model_idxes, batch_idxes]
                #print(samples.shape,samples)
                
                pred_reward, pred_obs = samples[:, :1], samples[:, 1:]
                #print(pred_reward.shape,pred_obs.shape)
        rollout = dict()
                #! trajectories are dict of keys {obs,actions,next_obs,rewards,dones,episode_dones}
        rollout['obs'] = to_np(sampled_batch['next_obs'])
        rollout['actions'] = to_np(next_action)
        rollout['next_obs'] = to_np(pred_obs)
        rollout['rewards'] = to_np(pred_reward)
        rollout['dones'] = np.zeros_like(pred_reward.cpu())
        rollout['episode_dones'] = np.zeros_like(pred_reward.cpu())
        replay_model.push_batch(**rollout)

    def update_parameters(self, memory1,updates,memory2=None,alpha=0.05,iter=0):
        if(iter==0): #! static rate
            sampled_batch1 = memory1.sample(int(self.batch_size*alpha))
            sampled_batch2 = memory2.sample(self.batch_size-int(self.batch_size*alpha))
        else:
            alpha=min(0.8,float(iter/self.max_iter_use_real_data))
            #print(f'{alpha*100}percentage data are collected from model buffer')
            sampled_batch1 = memory1.sample(max(1,int(self.batch_size*alpha)))
            sampled_batch2=memory2.sample(self.batch_size-max(1,int(self.batch_size*alpha)))
        sampled_batch={}
        for key in sampled_batch1:
            if not isinstance(sampled_batch1[key], dict) and sampled_batch1[key].ndim == 1:
                sampled_batch1[key] = sampled_batch1[key][..., None]
        for key in sampled_batch2:
            if not isinstance(sampled_batch2[key], dict) and sampled_batch2[key].ndim == 1:
                sampled_batch2[key] = sampled_batch2[key][..., None]
        permutation = list(np.random.permutation(self.batch_size))
        # for key in sampled_batch:
        #     if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
        #         sampled_batch[key] = sampled_batch[key][..., None]
        sampled_batch=merge_dict(sampled_batch1,sampled_batch2,permutation)


        sampled_batch = to_torch(
            sampled_batch, dtype='float32', device=self.device, non_blocking=True)

        with torch.no_grad():  # ! 这里为啥不用梯度
            next_action, next_log_prob = self.policy(
                sampled_batch['next_obs'], mode='all')[:2]
            q_next_target = self.target_critic(
                sampled_batch['next_obs'], next_action)
            min_q_next_target = torch.min(
                q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            q_target = sampled_batch['rewards'] + \
                (1 - sampled_batch['dones']) * self.gamma * min_q_next_target
        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(
            q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
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
        }