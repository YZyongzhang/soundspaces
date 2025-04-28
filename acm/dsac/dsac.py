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
from data.angle_data import Data
from net.avf import AVFNet



class AVNet(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim

        # self.model_path = '../../data/checkpoint/acmcheckpoint/avf_adddata_30000.pth'
        # self.model_path = './data/checkpoint/acmcheckpoint/avf_adddata_30000.pth'
        # self.model_path = 'acm/checkpoint/avnf_finnal_1.pth'
        self.avf = AVFNet(hid_dim=128, out_put=4, width_dim=128, height_dim=36).to(self.device)
        # self.avf.load_state_dict(torch.load(self.model_path))
        # self.avf.eval()

        # 双 Q 网络
        self.Q_net1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
        self.Q_net2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

        # 策略网络（离散 SAC 需要）
        self.policy_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

    def forward(self, audio, visual):
        audio  = audio.squeeze(0)
        visual = visual.squeeze(0)
        combinencode = self.avf(audio, visual)
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
            target_q = reward.squeeze(0).unsqueeze(1) + (1 - done) * 0.99 * next_value.unsqueeze(1)
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

def muti_data(path):
    result = list()
    thispath = [f'{path}/level0',f'{path}/level1',f'{path}/level2']
    files0 = [os.path.join(thispath[0] , i) for i in os.listdir(thispath[0])]
    files1 = [os.path.join(thispath[1] , i) for i in os.listdir(thispath[1])]
    files2 = [os.path.join(thispath[2] , i) for i in os.listdir(thispath[2])]
    result.extend(files0)
    result.extend(files1)
    result.extend(files2)
    return result

def train():
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVNet(128, 4, 128, 36).to(device)
    model.train()
    cql = DiscreteSAC_CQL(model, device)
    episode = 0
    
    
    # 数据集加载
    path = ['../../data/RL/success_and_stop_data/level0' ,'../../data/RL/success_and_stop_data/level1', '../../data/RL/success_and_stop_data/level2']
    split = 100
    data_file = list()
    files0 = [os.path.join(path[0] , i) for i in os.listdir(path[0])]
    files1 = [os.path.join(path[1] , i) for i in os.listdir(path[1])]
    files2 = [os.path.join(path[2] , i) for i in os.listdir(path[2])]
    split_lists_level1 = np.array_split(files1, split)
    split_lists_level2 = np.array_split(files2, split)
    split_lists_level1 = [list(sublist) for sublist in split_lists_level1]
    split_lists_level2 = [list(sublist) for sublist in split_lists_level2]
    # 比如前50个epoch是l0,50-150增加到l0-l1是1:1,150-250增加到l0:l1:l2=1:1:1;250-300就不变
    data_file.extend(files0)
    data_file.extend(files1)
    data_file.extend(files2)
    muti_env = '../../data/RL/muti_env_data'
    envs_data = [os.path.join(muti_env , i) for i in os.listdir(muti_env)]
    for i in envs_data:
        data_file.extend(muti_data(i))
    num_epochs = 1000
    
    
    dataset = Data(data_file)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for epoch in range(num_epochs):
        # if epoch < 50 :
        #     dataset = Data(data_file)
        #     dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        # elif epoch >= 50 and epoch < 150 :
        #     data_file.extend(split_lists_level1[epoch-50])
        #     dataset = Data(data_file)
        #     dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        # elif epoch >= 150 and epoch < 250 :
        #     data_file.extend(split_lists_level1[epoch-150])
        #     dataset = Data(data_file)
        #     dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        for batch_data in dataloader:
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_done, batch_reward, batch_labels = batch_data
            batch_done = torch.stack(batch_done).to(device)
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
                torch.save(model.state_dict(), f'./checkpoint/DSAC_mutienv_{episode}.pth')
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f'log : ./logs/loss/DSAC10 ; path : ./checkpoint/DSAC2.pth ; commit: batchsize = 32 ')
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    torch.save(model.state_dict(), './checkpoint/DSAC_mutienv.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/DSAC23'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC23.log', level=logging.INFO,filemode='a')
    train()
