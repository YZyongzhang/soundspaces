import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical 
import librosa
import numpy as np
import os
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import  pickle
import logging
import time
import copy
# 自定义 AVFNet
import pdb
import sys
from yz.config import agent_config

class Critic_Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        
        self.input_dim = 0
        self.hidden_dim = 0
        self.gru_layers = 0
        
        
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
        self.clip_grad_param = 1

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    def compute_loss(self, q1, q2, logits, policy_dist ,target_q, action):
        a_Q1 = q1.gather(1, action.unsqueeze(1))
        a_Q2 = q2.gather(1, action.unsqueeze(1))
        min_q = torch.min(q1,q2)
        # min_q = torch.min(a_Q1,a_Q2)
        critic1_loss = F.mse_loss(a_Q1 , target_q.unsqueeze(1))
        critic2_loss = F.mse_loss(a_Q2 , target_q.unsqueeze(1))
        
        # critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - q1.mean()) 
                                                 
        # critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - q2.mean())
        critic1_regularization = self.alpha_cql * (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) 
                                                 
        critic2_regularization = self.alpha_cql * (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean())
        
        # critic1_regularization = 0
                                                 
        # critic2_regularization = 0
        
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
 
class Critic_Actor_GRU(nn.Module):
    def __init__(self, action_dim, input_dim=128, hidden_dim=96, gru_layers=1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim

        # GRU layer
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=gru_layers, batch_first=True)

        # Critic networks
        self.Q_net1 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        self.Q_net2 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, combinencode , hidden = None):
        # Pass the sequence through the GRU
        
        batchsize , gru_sqe_len , _ = combinencode.shape
        gru_out, _ = self.gru(combinencode , hidden)
        
        x = gru_out.reshape(batchsize * gru_sqe_len , -1)
        # Compute Q-values and policy logits
        q1 = self.Q_net1(x)
        q2 = self.Q_net2(x)
        logits = self.policy_net(x)
        probs = self.softmax(logits)
        return q1, q2, probs

class DiscreteSAC_CQL_GRU:
    def __init__(self, model, target_model, device, learning_rate=1e-5, cql_alpha=0.1, tau=0.005):
        self.model = model
        self.target_entropy = -4 
        self.target_model = target_model
        
        self.device = device
        self.lr = learning_rate
        self.tau = tau 
        
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.alpha_cql = cql_alpha
        self.clip_grad_param = 1

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    def get_value(self , pre_state , next_state, labels, reward, done):
        q1 , q2 , probs = self.model(pre_state)
        with torch.no_grad():
            next_q1 , next_q2 ,next_policy = self.target_model(next_state)
            next_min_q = torch.min(next_q1, next_q2)
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=1)
            # pdb.set_trace()
            target_q = reward + (1 - done) * 0.99 * next_value.unsqueeze(1)
        return q1 , q2 ,  probs , target_q
    def compute_loss(self, q1, q2, policy_dist ,target_q, action):
        # pdb.set_trace()
        a_Q1 = q1.gather(1, action)
        a_Q2 = q2.gather(1, action)
        min_q = torch.min(q1,q2)
        min_aq = torch.min(a_Q1,a_Q2)
        # pdb.set_trace()
        critic_loss = F.mse_loss(min_aq , target_q)
        critic_regularization = (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) + \
                    (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean())
        
        
        policy_loss = (policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_aq)).sum(1).mean()
        
        total_loss = critic_loss + critic_regularization + policy_loss
        return total_loss , critic_loss ,critic_regularization ,policy_loss

    def optimize_loss(self, total_loss, alpha_loss):
        self.model_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        clip_grad_norm_(self.model.parameters(), self.clip_grad_param)
        self.model_optimizer.step()
          
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def soft_update(self, model, target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_step(self, pre_state, next_state, action, reward, done):
        batchsize , gru_sqe_len , _ = pre_state.shape
        reward = reward.reshape(batchsize*gru_sqe_len,-1)
        done = done.reshape(batchsize*gru_sqe_len,-1)
        action = action.reshape(batchsize*gru_sqe_len,-1)
        
        q1 , q2 ,policy_dist , target_q = self.get_value( pre_state , next_state, action, reward, done)

        total_loss , critic_loss ,critic_regularization ,policy_loss = self.compute_loss(q1, q2, policy_dist ,target_q , action)
        
        
        entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
        alpha_loss = torch.mean(-self.log_alpha.exp() * (entropy.detach() + self.target_entropy))
        self.optimize_loss(total_loss,alpha_loss)
        self.soft_update(self.model , self.target_model)
        
        self.alpha = self.log_alpha.exp()
        
        loss_dict = {
            'total_loss':total_loss.item(),
            'critic_loss':critic_loss.item(),
            'critic_regularization':critic_regularization,
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

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=64, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    def hidden_init(self,layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
    
class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device,
                        gru_inputsize=0,
                        gru_hidden_size=0,
                       
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 0.005
        hidden_size = 32
        learning_rate = 1e-5
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # GRU Network
        self.gru = nn.GRU(input_size=gru_inputsize, hidden_size=gru_hidden_size, num_layers=2, batch_first=True).to(device)
        self.gru_optimizer = optim.Adam(self.gru.parameters(), lr=learning_rate)
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        self.softmax = nn.Softmax(dim=-1)

    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        # pdb.set_trace()
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self,  experiences , epoch):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, next_states ,actions, rewards, dones = experiences
        # pdb.set_trace()
        batch_size , gru_time_seq , _ = states.shape
        actions = actions.reshape(batch_size*gru_time_seq,-1)
        rewards = rewards.reshape(batch_size*gru_time_seq,-1)
        dones   = dones.reshape(batch_size*gru_time_seq , -1)
        states,_ = self.gru(states)
        states = states.reshape(batch_size*gru_time_seq,-1)
        with torch.no_grad():
            next_states,_ = self.gru(next_states)
            next_states = next_states.reshape(batch_size*gru_time_seq,-1)
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        # pdb.set_trace()
        entropy = -log_pis.mean()
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        # Compute alpha loss
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)
            # pdb.set_trace()
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(1)) 


        # Compute critic loss
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        # pdb.set_trace()
        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())
        
        critic1_loss = F.mse_loss(q1_, Q_targets)
        critic2_loss = F.mse_loss(q2_, Q_targets)
        # pdb.set_trace()
        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + 0.1*cql1_scaled_loss
        total_c2_loss = critic2_loss + 0.1*cql2_scaled_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.gru_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        self.gru_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "cql1_scaled_loss": cql1_scaled_loss.item(),
            "cql2_scaled_loss": cql2_scaled_loss.item(),
            "current_alpha": current_alpha.item(),
            "cql_alpha_loss": cql_alpha_loss.item(),
            "cql_alpha": cql_alpha.item(),
            "entropy":entropy.item()
        }

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class AVNet(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim


        self.Q_net1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
        self.Q_net2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

    def forward(self, combinencode):
        q1 = self.Q_net1(combinencode)
        q2 = self.Q_net2(combinencode)
        logits = self.policy_net(combinencode)
        return q1, q2, logits
    
class DiscreteSAC_CQL_old(nn.Module):
    def __init__(self, device, learning_rate=1e-5, alpha=0.1, tau=0.005):
        super().__init__()
        self.model = Critic_Actor_GRU(action_dim=4).to(device)
        self.target_entropy = -4  # 目标熵（离散 SAC）
        self.target_model = Critic_Actor_GRU(action_dim=4).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.device = device
        self.lr = learning_rate
        self.tau = tau  # 目标网络软更新系数

        # alpha 相关参数
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)  # log_alpha 存储
        self.alpha = self.log_alpha.exp().detach() # 计算 alpha , 脱离计算图
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)  # Adam 优化器

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, q1, q2, logits, target_q, labels):
        """ 计算离散 SAC + CQL 损失 """
        # pdb.set_trace()
        # 选取执行的动作 Q 值
        # pdb.set_trace()
        a_Q1 = q1.gather(1, labels)
        a_Q2 = q2.gather(1, labels)
        min_aq = torch.min(a_Q1, a_Q2)  # 双 Q 学习
        
        min_q = torch.min(q1,q2)
        
        # Q-learning 目标
        # pdb.set_trace()
        q_loss = 0.5*F.mse_loss(min_aq, target_q)

        # CQL 额外约束
        q_regularization = (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) + \
                           (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean())

        # 策略损失（离散 SAC）
        policy_dist = F.softmax(logits, dim=1)
        policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_q), dim=1))

        total_loss = q_loss +  q_regularization + policy_loss
        return total_loss, q_loss, q_regularization, policy_loss 

    def train_step(self, pre_state , next_state, labels, reward, done):
        # input size (batch,timeseq,-1)
        batchsize , gru_sqe_len , _ = pre_state.shape
        reward = reward.reshape(batchsize*gru_sqe_len,-1)
        done = done.reshape(batchsize*gru_sqe_len,-1)
        labels = labels.reshape(batchsize*gru_sqe_len,-1)
        # pre_state = pre_state.squeeze(0)
        # next_state = next_state.squeeze(0)
        # labels = labels.squeeze(0)
        # reward = reward.squeeze(0)
        # done = done.squeeze(0)
        q1, q2, logits = self.model(pre_state)
        with torch.no_grad():
            next_q1, next_q2, next_logits = self.target_model(next_state)
            next_min_q = torch.min(next_q1, next_q2)
            next_policy = F.softmax(next_logits, dim=1)
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=-1)
            # pdb.set_trace()
            target_q = reward + (1 - done) * 0.99 * next_value.unsqueeze(1)
        total_loss, q_loss, q_regularization, policy_loss = self.compute_loss(q1, q2, logits, target_q, labels)
        
        # _,_, alp_logits = self.model(pre_audio, pre_visual)
        # 计算 entropy 并优化 alpha
        policy_dist = F.softmax(logits, dim=1)
        entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
        alpha_loss = -torch.mean(self.log_alpha.exp() * (entropy.detach() + self.target_entropy))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 更新 alpha
        self.alpha = self.log_alpha.exp()

        # 目标网络软更新
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
    "total_loss": total_loss.item(),
    "q_loss": q_loss.item(),
    "q_regularization": q_regularization.item(),
    "policy_loss": policy_loss.item(),
    "alpha_loss": alpha_loss.item(),
    "alpha": self.alpha.item(),
}
