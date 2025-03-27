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
from concurrent.futures import ProcessPoolExecutor
# 自定义 AVFNet
# from acm.dsac.data.data1 import Data
# from acm.dsac.net.avf import AVFNet
from data.angle_data import Data
from net.avf import AVFNet
# from acm.dsac.data.newdata import newMyData
class AVNet(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim

        self.model_path = '../checkpoint/avnf_finnal_1.pth'
        # self.model_path = 'acm/checkpoint/avnf_finnal_1.pth'
        self.avf = AVFNet(hid_dim=128, out_put=4, width_dim=128, height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.model_path))
        self.avf.eval()

        # 双 Q 网络
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

        # 策略网络（离散 SAC 需要）
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.initialize_weights_uniform()

    def forward(self, audio, visual):
        combinencode = self.avf(audio, visual)
        q1 = self.Q_net1(combinencode)
        q2 = self.Q_net2(combinencode)
        logits = self.policy_net(combinencode)
        action_logits = self.softmax(logits)
        return q1, q2, action_logits

    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
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
        a_Q1 = q1.gather(1, labels.unsqueeze(1))
        a_Q2 = q2.gather(1, labels.unsqueeze(1))
        min_q = torch.min(a_Q1, a_Q2)  # 双 Q 学习

        # Q-learning 目标
        q_loss = F.mse_loss(min_q, target_q)

        # CQL 额外约束
        q_regularization = (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) + \
                           (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean())

        # 策略损失（离散 SAC）
        policy_dist = F.softmax(logits, dim=1)
        policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_q), dim=1))

        total_loss = q_loss + 0.5 * q_regularization + policy_loss
        return total_loss, q_loss, q_regularization, policy_loss 

    def train_step(self, pre_audio, pre_visual, next_audio, next_visual, labels, reward, done):
        pre_audio, pre_visual, next_audio, next_visual, labels, reward, done = \
            pre_audio.float(), pre_visual.float(), next_audio.float(), next_visual.float(), \
            labels.long(), reward.float(), done.float()

        q1, q2, logits = self.model(pre_audio, pre_visual)
        with torch.no_grad():
            next_q1, next_q2, next_logits = self.target_model(next_audio, next_visual)
            next_min_q = torch.min(next_q1, next_q2)
            next_policy = F.softmax(next_logits, dim=1)
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=1)
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * 0.99 * next_value.unsqueeze(1)

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
    model.load_state_dict(torch.load('./checkpoint/DSAC_muti_level_30000.pth')) # 接着训练
    model.train()
    cql = DiscreteSAC_CQL(model, device)
    episode = 30000
    # 数据集加载
    path_ = ['../../data/RL/muti_level_data/level0' , '../../data/RL/muti_level_data/level1', '../../data/RL/muti_level_data/level2']
    level0 = [os.path.join(path_[0] , f) for f in os.listdir(path_[0])]
    level1 = [os.path.join(path_[1] , f) for f in os.listdir(path_[1])]
    level2 = [os.path.join(path_[2] , f) for f in os.listdir(path_[2])]
    num_split = 50
    split_lists = []
    split_lists_level0 = np.array_split(level0, num_split)
    split_lists_level1 = np.array_split(level1, num_split)
    split_lists_level2 = np.array_split(level2, num_split)
    split_lists_level0 = [list(sublist) for sublist in split_lists_level0]
    split_lists_level1 = [list(sublist) for sublist in split_lists_level1]
    split_lists_level2 = [list(sublist) for sublist in split_lists_level2]
    
    split_lists.extend(split_lists_level0)
    split_lists.extend(split_lists_level1)
    split_lists.extend(split_lists_level2)
    num_epochs = 500
    for epoch in range(100 , num_epochs):
        # if epoch == 50 :
        #     split_lists.extend(split_lists_level1)
        # if epoch == 100 :
        #     split_lists.extend(split_lists_level1)
        logging.info(f'split_lists len is {len(split_lists)}')
        for i in split_lists:
            dataset = Data(path=i)
            dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
            for batch_data in dataloader:
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_done, batch_reward, batch_labels = batch_data
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done = \
                    batch_pre_audio.to(device), batch_pre_visual.to(device), batch_next_audio.to(device), batch_next_visual.to(device), \
                    batch_labels.to(device), batch_reward.to(device), batch_done.to(device)

                total_loss, q_loss, q_regularization, policy_loss ,alpha_loss, alpha= cql.train_step(
                    batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done
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
                    torch.save(model.state_dict(), f'./checkpoint/DSAC_muti_level_{episode}.pth')
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f'log : ./logs/loss/DSAC12 ; path : ./checkpoint/DSAC_muti_level.pth ; commit: batchsize = 32 ')
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    torch.save(model.state_dict(), './checkpoint/DSAC_muti_level.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/DSAC12'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC12.log', level=logging.INFO,filemode='a')
    train()
