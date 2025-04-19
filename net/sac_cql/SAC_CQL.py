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
        a_Q1 = q1.gather(1, action.unsqueeze(1))
        a_Q2 = q2.gather(1, action.unsqueeze(1))
        min_q = torch.min(q1,q2)
        critic1_loss = F.mse_loss(a_Q1 , target_q.unsqueeze(1))
        critic2_loss = F.mse_loss(a_Q2 , target_q.unsqueeze(1))
        
        # critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - q1.mean()) 
                                                 
        # critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - q2.mean())
        critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) 
                                                 
        critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean())
        
        critic1_regularization = 0
                                                 
        critic2_regularization = 0
        
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
                'critic1_regularization':critic1_regularization,
                'critic2_regularization':critic2_regularization,
                'policy_loss':policy_loss.item(),
                'alpha_loss':alpha_loss.item(),
                'alpha':self.alpha.item(),
                'entropy':entropy.item()
            }
        return loss_dict

class AI_DiscreteSAC_CQL:
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
    def compute_loss(self, q1, q2, logits, policy_dist, target_q, action):
            # 选中执行的动作的 Q 值
            a_Q1 = q1.gather(1, action.unsqueeze(1))
            a_Q2 = q2.gather(1, action.unsqueeze(1))
            
            critic1_loss = F.mse_loss(a_Q1, target_q.unsqueeze(1))
            critic2_loss = F.mse_loss(a_Q2, target_q.unsqueeze(1))

            # CQL regularization：最大化 Q(s, a) gap（logsumexp - 真实 Q）
            logsumexp_q1 = torch.logsumexp(q1, dim=1).mean()
            logsumexp_q2 = torch.logsumexp(q2, dim=1).mean()
            q1_mean = a_Q1.mean()
            q2_mean = a_Q2.mean()
            critic1_regularization = self.alpha_cql * (logsumexp_q1 - q1_mean)
            critic2_regularization = self.alpha_cql * (logsumexp_q2 - q2_mean)

            # policy loss: soft Q learning 损失，鼓励采样低值动作
            min_q = torch.min(q1, q2)
            log_prob = torch.log(policy_dist + 1e-10)
            policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * log_prob - min_q), dim=1))

            return critic1_loss, critic2_loss, critic1_regularization, critic2_regularization, policy_loss

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
    def train_step(self, pre_state, next_state, action, reward, done):
        with torch.no_grad():
            q1_next = self.target_critic1(next_state)
            q2_next = self.target_critic2(next_state)
            next_logits = self.policy(next_state)
            next_policy = F.softmax(next_logits, dim=1)
            next_log_policy = torch.log(next_policy + 1e-10)
            min_q_next = torch.min(q1_next, q2_next)
            next_value = (next_policy * (min_q_next - self.alpha.detach() * next_log_policy)).sum(dim=1)
            target_q = reward + (1 - done) * 0.99 * next_value

        q1 = self.critic1(pre_state)
        q2 = self.critic2(pre_state)
        logits = self.policy(pre_state)
        policy_dist = F.softmax(logits, dim=1)

        critic1_loss, critic2_loss, cql1_reg, cql2_reg, policy_loss = self.compute_loss(
            q1, q2, logits, policy_dist, target_q, action)

        # Total loss
        total_critic1_loss = critic1_loss + cql1_reg
        total_critic2_loss = critic2_loss + cql2_reg

        # Entropy
        entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
        alpha_loss = -(self.log_alpha.exp() * (entropy + self.target_entropy).detach()).mean()

        # 优化器 step
        self.optimize_loss(policy_loss, total_critic1_loss, total_critic2_loss, alpha_loss)

        # soft update
        self.soft_updata(self.critic1, self.target_critic1)
        self.soft_updata(self.critic2, self.target_critic2)
        self.alpha = self.log_alpha.exp()

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'critic1_regularization': cql1_reg.item(),
            'critic2_regularization': cql2_reg.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'entropy': entropy.item()
        }
              
class CQLSAC_Improved:
    def __init__(self, model , target_model, device,
                 gamma=0.99, tau=0.005,
                 target_entropy=None, cql_temp=1.0,
                 cql_alpha_init=1.0, min_q_weight=5.0,
                 max_q_backup=True, clip_grad_norm=5.0):

        self.model = model
        self.lr = 1e-5
        action_dim = 4
        self.target_model = target_model
        self.actor = self.model.policy_net
        self.critic1 = self.model.Q_net1
        self.critic2 = self.model.Q_net2
        self.target_critic1 = self.target_model.Q_net1
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = self.target_model.Q_net2
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.target_entropy = target_entropy if target_entropy is not None else -torch.log(torch.tensor(1.0 / action_dim)).item()

        self.cql_temp = cql_temp
        self.cql_log_alpha = torch.tensor([cql_alpha_init], requires_grad=True, device=device)
        self.min_q_weight = min_q_weight
        self.max_q_backup = max_q_backup
        
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.cql_alpha_optimizer = optim.Adam([self.cql_log_alpha], lr=3e-4)
        self.clip_grad_norm = clip_grad_norm

    def train_step(self, obs , next_obs ,actions , rewards , dones):
        # Compute alpha loss
        with torch.no_grad():
            logits = self.actor(obs)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp().clamp(min=1e-6)

        # Compute target Q
        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            target_q1 = self.target_critic1(next_obs)
            target_q2 = self.target_critic2(next_obs)
            target_q = torch.min(target_q1, target_q2)

            target = (next_probs * (target_q - alpha * next_log_probs)).sum(dim=-1)
            target = rewards + self.gamma * (1.0 - dones) * target

        # Critic loss
        current_q1 = self.critic1(obs).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        current_q2 = self.critic2(obs).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        critic1_loss = F.mse_loss(current_q1, target.detach())
        critic2_loss = F.mse_loss(current_q2, target.detach())


        # CQL Regularization

        q1_vals = self.critic1(obs)
        q2_vals = self.critic2(obs)

        logsumexp1 = torch.logsumexp(q1_vals / self.cql_temp, dim=-1)
        logsumexp2 = torch.logsumexp(q2_vals / self.cql_temp, dim=-1)
        cql1_loss = (logsumexp1 - q1_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)).mean()
        cql2_loss = (logsumexp2 - q2_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)).mean()

        cql_alpha = F.softplus(self.cql_log_alpha).clamp(max=10.0)
        cql_loss = cql_alpha * (cql1_loss + cql2_loss)

        total_critic1_loss = critic1_loss + self.min_q_weight * cql1_loss
        total_critic2_loss = critic2_loss + self.min_q_weight * cql2_loss
        
        
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_critic1_loss.backward()
        total_critic2_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_norm)
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_norm)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Actor Loss
        new_logits = self.actor(obs)
        new_probs = F.softmax(new_logits, dim=-1)
        new_log_probs = F.log_softmax(new_logits, dim=-1)

        q1 = self.critic1(obs)
        q2 = self.critic2(obs)
        min_q = torch.min(q1, q2)
        actor_loss = (new_probs * (alpha * new_log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()

        # CQL Alpha Loss
        cql_alpha_loss = -(self.cql_log_alpha * (cql1_loss + cql2_loss).detach()).mean()
        self.cql_alpha_optimizer.zero_grad()
        cql_alpha_loss.backward()
        self.cql_alpha_optimizer.step()

        # Soft update
        with torch.no_grad():
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

        return {
            'loss/critic1_loss': critic1_loss.item(),
            'loss/critic2_loss': critic2_loss.item(),
            'loss/policy_loss': actor_loss.item(),
            'loss/alpha': alpha.item(),
            'loss/alpha_loss': alpha_loss.item(),
            'loss/cql_alpha': cql_alpha.item(),
            'loss/cql_alpha_loss': cql_alpha_loss.item(),
            'loss/critic1_regularization': cql1_loss.item(),
            'loss/critic2_regularization': cql2_loss.item(),
            'loss/entropy': entropy.item(),
        }
        
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
        # min_q = torch.min(q1,q2)
        min_q = torch.min(a_Q1 , a_Q2)
        critic1_loss = F.mse_loss(a_Q1 , target_q.unsqueeze(1))
        critic2_loss = F.mse_loss(a_Q2 , target_q.unsqueeze(1))
        
        # critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - q1.mean()) 
                                                 
        # critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - q2.mean())
        
        critic1_regularization = 0 
                                                 
        critic2_regularization = 0
        
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
                'critic1_regularization':critic1_regularization,
                'critic2_regularization':critic2_regularization,
                'policy_loss':policy_loss.item(),
                'alpha_loss':alpha_loss.item(),
                'alpha':self.alpha.item(),
                'entropy':entropy.item()
            }
        return loss_dict