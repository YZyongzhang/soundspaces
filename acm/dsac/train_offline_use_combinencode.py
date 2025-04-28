import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
import os
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
import time
from concurrent.futures import ProcessPoolExecutor
# 自定义 AVFNet
import pdb
from data.use_combinencode import Data



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
    
class DiscreteSAC_CQL:
    def __init__(self, model, device, learning_rate=1e-4, alpha=0.1, tau=0.005):
        self.model = model
        self.target_entropy = -4  # 目标熵（离散 SAC）
        self.target_model = AVNet(128, 4, 128, 36).to(device)
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

        # 选取执行的动作 Q 值
        a_Q1 = q1.gather(1, labels.squeeze(0).unsqueeze(1))
        a_Q2 = q2.gather(1, labels.squeeze(0).unsqueeze(1))
        min_q = torch.min(a_Q1, a_Q2)  # 双 Q 学习

        # Q-learning 目标
        # pdb.set_trace()
        q_loss = F.mse_loss(min_q, target_q.unsqueeze(1))

        # CQL 额外约束
        q_regularization = (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) + \
                           (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean())

        # 策略损失（离散 SAC）
        policy_dist = F.softmax(logits, dim=1)
        policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_q), dim=1))

        total_loss = q_loss + 0.5 * q_regularization + policy_loss
        return total_loss, q_loss, q_regularization, policy_loss 

    def train_step(self, pre_state , next_state, labels, reward, done):


        q1, q2, logits = self.model(pre_state)
        with torch.no_grad():
            next_q1, next_q2, next_logits = self.target_model(next_state)
            next_min_q = torch.min(next_q1, next_q2)
            next_policy = F.softmax(next_logits, dim=1)
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=1)
            # pdb.set_trace()
            target_q = reward + (1 - done) * 0.99 * next_value
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

        return total_loss.item(), q_loss.item(), q_regularization.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item()


def train():
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVNet(128, 4, 128, 36).to(device)
    model.train()
    cql = DiscreteSAC_CQL(model, device)
    episode = 0
    
    
    # 数据集加载
    database = '../../data/database/combinencode_level/combinencode_0'
    dataset = Data(database)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    num_epochs = 500
    for epoch in range(num_epochs):
        for batch_data in dataloader:
            batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch_data
            batch_done = batch_done.to(device)
                
            total_loss, q_loss, q_regularization, policy_loss ,alpha_loss, alpha= cql.train_step(
                batch_pre_state ,batch_next_state, batch_labels, batch_reward, batch_done
            )
            writer.add_scalar('Loss/total_loss', total_loss, episode)
            writer.add_scalar('Loss/q_loss', q_loss, episode)
            writer.add_scalar('Loss/q_regularization', q_regularization, episode)
            writer.add_scalar('Loss/policy_loss', policy_loss, episode)
            writer.add_scalar('Loss/alpha_loss', alpha_loss, episode)
            writer.add_scalar('Loss/alpha', alpha, episode)
            print(f'Epoch {epoch} , episode : {episode}: Loss {total_loss}, Q-Loss {q_loss}, CQL-Reg {q_regularization}, Policy Loss {policy_loss}')
            
            
            if episode % 1000 == 0 :
                logging.info(f"Epoch {epoch} , episode : {episode} ,time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
            episode+=1
            if episode % 10000 == 0 and episode != 0:
                torch.save(model.state_dict(), f'./checkpoint/DSAC_mutienv_combinencode_0_{episode}.pth')
                
                
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f'log : ./logs/loss/DSAC10 ; path : ./checkpoint/DSAC2.pth ; commit: batchsize = 32 ')
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    torch.save(model.state_dict(), './checkpoint/DSAC_mutienv_combinencode_0.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/DSAC_use_combinencode_0'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC_use_combinencode_0.log', level=logging.INFO,filemode='a')
    train()
