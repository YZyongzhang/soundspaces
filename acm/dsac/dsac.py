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
from acm.dsac.data.data1 import Data
from acm.dsac.net.avf import AVFNet
# from data.data1 import Data
# from net.avf import AVFNet
# from acm.dsac.data.newdata import newMyData
class AVNet(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim

        # self.model_path = '../checkpoint/avnf_finnal_1.pth'
        self.model_path = 'acm/checkpoint/avnf_finnal_1.pth'
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
# class MyData(Dataset):
#     def __init__(self,path):
#         batch_data = self.load_data(path=path)
#         # batch_data shape (batch , list)
#         # result = [seq_list , {
#         #     'path_point':self.path_point,
#         #     'sound_pos':env.get_source_pos()[0]
#         # } , done ,self.obs]
#         self.preaudio,self.previsual, self.nextaudio, self.nextvisual,self.done,self.reward,self.action = self.deal_data(batch_data)
#     def __len__(self):
#         return len(self.preaudio)
#     def __getitem__(self,idx):
#         return self.preaudio[idx],self.previsual[idx], self.nextaudio[idx], self.nextvisual[idx],self.done[idx],self.reward[idx],self.action[idx]
#     def load_data(self,path):
#         # 将整个数据打包成一个大的batch
#         files = list()
#         for p in path:
#             files_path = os.listdir(p)
#             for f in files_path[:10]:
#                 files.append(os.path.join(p,f))
#         batch_data = list()
#         for i in tqdm(files,desc="load files" , unit='file'):
#             with open(i , 'rb') as f:
#                 data_ = pickle.load(f)
#             batch_data.append(data_)
#         return batch_data
#     def get_data(self , data , name):
#         d = list()
#         for i in range(len(data)):
#             d.extend(data[i][name])
#         return d
#     def get_error(self):
#         return self.error
#     def deal_data(self,batch_datas):
#         pre_audio = list()
#         pre_visual = list()
#         next_audio = list()
#         next_visual = list()
#         done = list()
#         reward = list()
#         action = list()
#         self.error = list()
#         for batch_id,batch_data in enumerate(tqdm(batch_datas,desc='deal batch_data' , unit="batch_datas")):
#             data = batch_data[0]
#             done_ = batch_data[2]
#             frist_state = batch_data[3]
#             if len(data) != 1:
#                 continue
#             self.audio_ = self.get_data(data,'audio')
#             self.tag_ = self.get_data(data,'rl_pred')
#             self.visual_ = self.get_data(data,'camera')
#             self.reward_ = self.get_data(data,'reward')
#             self.done = list()
#             self.next_audio = list()
#             self.tag = list()
#             self.next_visual = list()
#             self.reward = list()
#             for index,tag in enumerate(self.tag_):
#                 if tag == 3 or index == 199:
#                     self.next_audio = self.audio_[:index+1]
#                     self.next_visual = self.visual_[:index+1]
#                     self.reward = self.reward_[:index+1]
#                     self.tag = self.tag_[:index+1]
#                     # 不包含stop信息，如需包含请index+1。具体就是指最后执行完stop之后不再会有相同的img产生
#                     break
#             self.next_audio.pop(0)
#             self.next_visual.pop(0)
#             self.reward.pop(0)
#             self.tag.pop(0)
#             for d in done_:
#                 if d[0] == False :
#                     self.done.append(0)
#                 elif d[0] == True:
#                     self.done.append(1)
#             ##  get state , next_state
#             self.pre_audio = [frist_state[0]['audio']] + self.next_audio[:-1]
#             self.pre_visual = [frist_state[0]['camera']] + self.next_visual[:-1]
#             if len(self.done) != len(self.tag):
#                 self.error.append(batch_id)
#             pre_audio.extend(self.pre_audio)
#             pre_visual.extend(self.pre_visual)
#             next_audio.extend(self.next_audio)
#             next_visual.extend(self.next_visual)
#             done.extend(self.done)
#             reward.extend(self.reward)
#             action.extend(self.tag)
#         return pre_audio,pre_visual, next_audio, next_visual,done,reward,action


class MyData(Dataset):
    def __init__(self, path):
        batch_data = self.load_data(path=path)
        # batch_data shape (batch , list)
        # result = [seq_list , {
        #     'path_point':self.path_point,
        #     'sound_pos':env.get_source_pos()[0]
        # } , done ,self.obs]
        self.preaudio, self.previsual, self.nextaudio, self.nextvisual, self.done, self.reward, self.action = self.deal_data(batch_data)

    def __len__(self):
        return len(self.preaudio)

    def __getitem__(self, idx):
        return self.preaudio[idx], self.previsual[idx], self.nextaudio[idx], self.nextvisual[idx], self.done[idx], self.reward[idx], self.action[idx]

    def load_data(self, path):

        batch_data = []
        with ProcessPoolExecutor() as executor:
            batch_data = list(tqdm(executor.map(self._load_file, path), desc="Loading files", unit='file'))
        
        return batch_data

    def _load_file(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_data(self, data, name):
        d = []
        for item in data:
            d.extend(item[name])
        return d

    def get_error(self):
        return self.error

    def deal_data(self, batch_datas):
        pre_audio, pre_visual, next_audio, next_visual = [], [], [], []
        done, reward, action = [], [], []
        self.error = []

        for batch_id, batch_data in enumerate(tqdm(batch_datas, desc='Processing batch_data', unit="batch")):
            data, done_, first_state = batch_data[0], batch_data[2], batch_data[3]

            if len(data) != 1:
                continue

            self.audio_ = self.get_data(data, 'audio')
            self.tag_ = self.get_data(data, 'rl_pred')
            self.visual_ = self.get_data(data, 'camera')
            self.reward_ = self.get_data(data, 'reward')

            # Initialize lists for current batch
            current_done, current_next_audio, current_tag, current_next_visual, current_reward = [], [], [], [], []

            for index, tag in enumerate(self.tag_):
                if tag == 3 or index == 199:
                    current_next_audio = self.audio_[:index + 1]
                    current_next_visual = self.visual_[:index + 1]
                    current_reward = self.reward_[:index + 1]
                    current_tag = self.tag_[:index + 1]
                    break

            # Removing the first element (since it's the next state)
            current_next_audio.pop(0)
            current_next_visual.pop(0)
            current_reward.pop(0)
            current_tag.pop(0)

            # Update done list
            current_done = [1 if d[0] else 0 for d in done_]

            # Get state and next_state
            self.pre_audio = [first_state[0]['audio']] + current_next_audio[:-1]
            self.pre_visual = [first_state[0]['camera']] + current_next_visual[:-1]

            if len(current_done) != len(current_tag):
                self.error.append(batch_id)

            # Efficiently extend lists using append
            pre_audio.extend(self.pre_audio)
            pre_visual.extend(self.pre_visual)
            next_audio.extend(current_next_audio)
            next_visual.extend(current_next_visual)
            done.extend(current_done)
            reward.extend(current_reward)
            action.extend(current_tag)

        return pre_audio, pre_visual, next_audio, next_visual, done, reward, action

    
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
    model.train()
    cql = DiscreteSAC_CQL(model, device)
    episode = 0
    # 数据集加载
    path = ['../../data/RL/easydata']
    paths = os.listdir(path=path[0])
    path_files = list()
    for i in paths:
        path_files.append(os.path.join(path[0] , i))
    num_split = 50
    split_lists = np.array_split(path_files, num_split)
    split_lists = [list(sublist) for sublist in split_lists]

    num_epochs = 500
    for epoch in range(num_epochs):
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
                    torch.save(model.state_dict(), f'./checkpoint/DSAC_easy_{episode}.pth')
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f'log : ./logs/loss/DSAC10 ; path : ./checkpoint/DSAC2.pth ; commit: batchsize = 32 ')
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
        torch.save(model.state_dict(), './checkpoint/DSAC_easy.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/DSAC11'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC11.log', level=logging.INFO,filemode='a')
    logging.info("使用简单的数据进行的训练，看看结果是怎么样的")
    train()
