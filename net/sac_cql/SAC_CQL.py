import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import  pickle
import logging
import time
# 自定义 AVFNet
import pdb
import sys
from yz.config import agent_config

class Critic_Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim


        self.Q_net1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        self.Q_net2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def forward(self, combinencode):
        q1 = self.Q_net1(combinencode)
        q2 = self.Q_net2(combinencode)
        logits = self.policy_net(combinencode)
        return q1, q2, logits
 
class DiscreteSAC_CQL:
    def __init__(self, model , target_model , device, learning_rate=1e-4, cql_alpha=0.1, tau=0.005):
        self.model = model
        self.critic1 = self.model.Q_net1
        self.critic2 = self.model.Q_net2
        self.policy = self.model.policy_net
        self.target_entropy = -4 
        self.target_model = target_model
        self.target_critic1 = self.target_model.Q_net1
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = self.target_model.Q_net2
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.device = device
        self.lr = learning_rate
        self.tau = tau 
        
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.alpha_cql = cql_alpha

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    def compute_loss(self, q1, q2, logits, policy_dist ,target_q, action):

        a_Q1 = q1.gather(1, action.squeeze(0).unsqueeze(1))
        a_Q2 = q2.gather(1, action.squeeze(0).unsqueeze(1))
        min_q = torch.min(q1,q2)
        critic1_loss = F.mse_loss(a_Q1 , target_q.unsqueeze(1))
        critic2_loss = F.mse_loss(a_Q2 , target_q.unsqueeze(1))
        
        critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - q1.mean()) 
                                                 
        critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - q2.mean())
        
        policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_q), dim=1))

        return critic1_loss , critic2_loss ,critic1_regularization , critic2_regularization ,policy_loss
    def get_value(self , pre_state , next_state, labels, reward, done):
        q1 = self.critic1(pre_state)
        q2 = self.critic2(pre_state)
        logits = self.policy(pre_state)
        with torch.no_grad():
            next_q1 = self.target_critic1(next_state)
            next_q2 = self.target_critic2(next_state)
            next_logits = self.policy(next_state)
            next_min_q = torch.min(next_q1, next_q2)
            next_policy = F.softmax(next_logits, dim=1)
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=1)
            target_q = reward + (1 - done) * 0.99 * next_value
        return q1 , q2 , logits , target_q
    def optimize_loss(self,policy_loss,total_critic1_loss,total_critic2_loss,alpha_loss):
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        
        self.critic1_optimizer.zero_grad()
        total_critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        total_critic2_loss.backward()
        self.critic2_optimizer.step()
        
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    def soft_updata(self,model ,target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def train_step(self, pre_state , next_state, action, reward, done):
        with torch.autograd.set_detect_anomaly(True):
            q1 , q2 ,logits , target_q = self.get_value( pre_state , next_state, action, reward, done)
            policy_dist = F.softmax(logits, dim=1)
            critic1_loss , critic2_loss ,critic1_regularization , critic2_regularization ,policy_loss = self.compute_loss(q1, q2, logits, policy_dist ,target_q , action)
            
            total_critic1_loss = critic1_loss + critic1_regularization
            total_critic2_loss = critic2_loss + critic2_regularization
            
            entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
            alpha_loss = torch.mean(-self.log_alpha.exp() * (entropy.detach() + self.target_entropy))
            
            self.optimize_loss(policy_loss,total_critic1_loss,total_critic2_loss,alpha_loss)
            self.soft_updata(self.critic1 , self.target_critic1)
            self.soft_updata(self.critic2 , self.target_critic2)
            
            self.alpha = self.log_alpha.exp()
            
            loss_dict = {
                'critic1_loss':critic1_loss.item(),
                'critic2_loss':critic2_loss.item(),
                'critic1_regularization':critic1_regularization.item(),
                'critic2_regularization':critic2_regularization.item(),
                'policy_loss':policy_loss.item(),
                'alpha_loss':alpha_loss.item(),
                'alpha':self.alpha.item(),
                'entropy':entropy.item()
            }
        return loss_dict
    
class lambda_DiscreteSAC_CQL:
    def __init__(self, model , target_model , device, learning_rate=1e-4, cql_alpha=0.1, tau=0.005):
        self.model = model
        self.critic1 = self.model.Q_net1
        self.critic2 = self.model.Q_net2
        self.policy = self.model.policy_net
        self.target_entropy = -4 
        self.target_model = target_model
        self.target_critic1 = self.target_model.Q_net1
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = self.target_model.Q_net2
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.device = device
        self.lr = learning_rate
        self.tau = tau 
        
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.alpha_cql = cql_alpha
        self.clip_grad_param = 1
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    def compute_loss(self, q1, q2, logits, policy_dist ,target_q, action):

        a_Q1 = q1.gather(1, action.squeeze(0).unsqueeze(1))
        a_Q2 = q2.gather(1, action.squeeze(0).unsqueeze(1))
        min_q = torch.min(q1,q2)
        critic1_loss = F.mse_loss(a_Q1 , target_q.unsqueeze(1))
        critic2_loss = F.mse_loss(a_Q2 , target_q.unsqueeze(1))
        
        critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - q1.mean()) 
                                                 
        critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - q2.mean())
        
        policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_q), dim=1))

        return critic1_loss , critic2_loss ,critic1_regularization , critic2_regularization ,policy_loss
    def get_value(self , pre_state , next_state, labels, reward, done):
        q1 = self.critic1(pre_state)
        q2 = self.critic2(pre_state)
        logits = self.policy(pre_state)
        with torch.no_grad():
            next_q1 = self.target_critic1(next_state)
            next_q2 = self.target_critic2(next_state)
            next_logits = self.policy(next_state)
            next_min_q = torch.min(next_q1, next_q2)
            next_policy = F.softmax(next_logits, dim=1)
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=1)
            target_q = reward + (1 - done) * 0.99 * next_value
        return q1 , q2 , logits , target_q
    def optimize_loss(self,policy_loss,total_critic1_loss,total_critic2_loss,alpha_loss):
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        
        self.critic1_optimizer.zero_grad()
        total_critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        total_critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    def soft_updata(self,model ,target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def train_step(self, pre_state , next_state, action, reward, done):
        with torch.autograd.set_detect_anomaly(True):
            q1 , q2 ,logits , target_q = self.get_value( pre_state , next_state, action, reward, done)
            policy_dist = F.softmax(logits, dim=1)
            critic1_loss , critic2_loss ,critic1_regularization , critic2_regularization ,policy_loss = self.compute_loss(q1, q2, logits, policy_dist ,target_q , action)
            
            total_critic1_loss = critic1_loss + critic1_regularization
            total_critic2_loss = critic2_loss + critic2_regularization
            
            entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
            alpha_loss = torch.mean(-self.log_alpha.exp() * (entropy.detach() + self.target_entropy))
            
            self.optimize_loss(policy_loss,total_critic1_loss,total_critic2_loss,alpha_loss)
            self.soft_updata(self.critic1 , self.target_critic1)
            self.soft_updata(self.critic2 , self.target_critic2)
            
            self.alpha = self.log_alpha.exp()
            
            loss_dict = {
                'critic1_loss':critic1_loss.item(),
                'critic2_loss':critic2_loss.item(),
                'critic1_regularization':critic1_regularization.item(),
                'critic2_regularization':critic2_regularization.item(),
                'policy_loss':policy_loss.item(),
                'alpha_loss':alpha_loss.item(),
                'alpha':self.alpha.item(),
                'entropy':entropy.item()
            }
        return loss_dict