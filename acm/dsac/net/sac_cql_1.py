"""
in this case , i attempt to use static cql-alpha 
and impove conventional loss 
J(Q)=E s∼D[αlog a∑exp(Q(s,a))−E a∼π β(a∣s)[Q(s,a)]]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torch.distributions import Categorical 
from torch.nn.utils import clip_grad_norm_
import os
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
import time
# from net.network import Actor , Critic , LSTM , AVF
from acm.dsac.net.network import Actor , Critic , LSTM
class DiscreteSAC_CQL_1(nn.Module):
    def __init__(self, learning_rate=1e-4, alpha=0.1, tau=0.005):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_entropy = torch.tensor(-4,device=self.device)  # 目标熵（离散 SAC）
        self.tau = tau  # 目标网络软更新系数
        self.lr = learning_rate
        self.gamma = 0.99
        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)
        
        # self.AVF  = AVF(128 , 4 ,128 , 36).to(self.device)
        self.lstm = LSTM(128, 64 ,1) 
        self.policy = Actor(64, 4, 128, 36).to(self.device)
        
        self.critic1 = Critic(64, 4, 128, 36).to(self.device)
        self.critic2 = Critic(64, 4, 128, 36).to(self.device)
        
        self.target_critic1 = Critic(64, 4, 128, 36).to(self.device) 
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        
        self.target_critic2 = Critic(64,4,128,36).to(self.device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # alpha 相关参数
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)  # log_alpha 存储
        self.alpha = self.log_alpha.exp().detach() # 计算 alpha , 脱离计算图
        # self.alpha = alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)  # Adam 优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(),lr = self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(),lr = self.lr)
        
    
    def policy_loss(self,state, current_alpha):
        _ , action_probs , action_logprobs = self.policy.evaluate(state)
        Q1 = self.critic1(state)
        Q2 = self.critic2(state)
        minQ = torch.min(Q1,Q2)
        # policy_loss = (action_probs * (current_alpha.to(self.device) * action_logprobs - minQ)).sum(1).mean()
        policy_loss = (action_probs * (current_alpha * action_logprobs - minQ)).sum(1).mean()
        log_action_pi = torch.sum(action_logprobs * action_probs, dim=1)
        return policy_loss, log_action_pi
    
    def get_Q_target(self,next_state ,rewards , dones,current_alpha):
        with torch.no_grad():
            _, action_probs, log_pis = self.policy.evaluate(next_state)
            Q_target1_next = self.target_critic1(next_state)
            Q_target2_next = self.target_critic2(next_state)
            # Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - current_alpha.to(self.device) * log_pis)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - current_alpha * log_pis)
            # Compute Q targets for current states (y_i)
            # use action_probs to multiplication ,so target_q should use sum(dim = 1)
            Q_targets = rewards+ (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(1))
        return Q_targets
    def train_step(self, pre_audio, pre_visual, next_audio, next_visual, labels, reward, done):
        self.policy.train()
        self.critic1.train()
        self.critic2.train()
        pre_audio, pre_visual, next_audio, next_visual, labels, reward, done = \
            pre_audio.float(), pre_visual.float(), next_audio.float(), next_visual.float(), \
            labels.long(), reward.float(), done.float()
        
        current_alpha = self.alpha
        # 首先求LSTM的输出：
        pre_state = self.lstm(pre_audio , pre_visual)
        with torch.no_grad():
            next_state = self.lstm(next_audio , next_visual)
        # 首先求actorloss
        policy_loss , log_action_pi  = self.policy_loss(pre_state,current_alpha=current_alpha)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        
        # 求alphaloss
        alpha_loss = - (self.log_alpha.exp() * (log_action_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # 重新设置self.alpha
        self.alpha = self.log_alpha.exp().detach()
        
        # 求criticloss
        
        Q_target = self.get_Q_target(next_state ,rewards=reward , dones=done , current_alpha=current_alpha)
        q1 = self.critic1(pre_state)
        q2 = self.critic2(pre_state)
        
        q1_ = q1.gather(1, labels)
        q2_ = q2.gather(1, labels)
        critic1_loss =  F.mse_loss(q1_, Q_target)
        critic2_loss =  F.mse_loss(q2_, Q_target)
        
        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1_.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2_.mean()
        
        total_c1_loss = critic1_loss + 0.5*cql1_scaled_loss
        total_c2_loss = critic2_loss + 0.5*cql2_scaled_loss
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True) # retain_graph = True to use target_q again
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        self.critic2_optimizer.step()
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        return {
            "policy_loss":policy_loss.item(), 
            "critic1_loss":critic1_loss.item(), 
            "critic2_loss":critic2_loss.item(),  
            "cql1_scaled_loss":cql1_scaled_loss.item(), 
            "cql2_scaled_loss":cql2_scaled_loss.item(), 
            "total_c1_loss":total_c1_loss.item(),
            "total_c2_loss":total_c2_loss.item(),
            "current_alpha":current_alpha,
            "alpha_loss":alpha_loss.item(),
            }
    def val_step(self , pre_audio , pre_visual):
        self.policy.eval()
        self.critic1.eval()
        self.critic2.eval()
        policy_reward = self.policy(pre_audio , pre_visual).max(-1)[0].sum()
        critic1_reward = self.critic1(pre_audio , pre_visual).max(-1)[0].sum()
        critic2_reward = self.critic2(pre_audio , pre_visual).max(-1)[0].sum()
        return {"policy_reward":policy_reward , "critic1_reward":critic1_reward , "critic2_reward":critic2_reward}
    def soft_update(self, local_model , target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)