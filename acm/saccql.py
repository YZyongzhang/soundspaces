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

# 自定义 AVFNet
# from acm.avf import AVFNet
from avf import AVFNet
class Critic(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim

        # self.model_path = './acm/checkpoint/avnf_finnal_1.pth'
        self.model_path = './checkpoint/avnf_finnal_1.pth'
        self.avf = AVFNet(hid_dim=128, out_put=4, width_dim=128, height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.model_path))
        self.avf.eval()

        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
    def forward(self, audio, visual):
        combinencode = self.avf(audio, visual)
        q = self.critic(combinencode)
        return q
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
class Actor(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim

        # self.model_path = './acm/checkpoint/avnf_finnal_1.pth'
        self.model_path = './checkpoint/avnf_finnal_1.pth'
        self.avf = AVFNet(hid_dim=128, out_put=4, width_dim=128, height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.model_path))
        self.avf.eval()
        
        self.actor_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.initialize_weights_uniform()

    def forward(self, audio, visual):
        combinencode = self.avf(audio, visual)
        action_probs = self.softmax(self.actor_net(combinencode))
        dist = Categorical(action_probs)
        action = dist.sample().to(self.device)
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action_probs, log_action_probabilities

    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
                    
class MyData(Dataset):
    def __init__(self,path):
        batch_data = self.load_data(path=path)
        # batch_data shape (batch , list)
        # result = [seq_list , {
        #     'path_point':self.path_point,
        #     'sound_pos':env.get_source_pos()[0]
        # } , done ,self.obs]
        self.preaudio,self.previsual, self.nextaudio, self.nextvisual,self.done,self.reward,self.action = self.deal_data(batch_data)
    def __len__(self):
        return len(self.preaudio)
    def __getitem__(self,idx):
        return self.preaudio[idx],self.previsual[idx], self.nextaudio[idx], self.nextvisual[idx],self.done[idx],self.reward[idx],self.action[idx]
    def load_data(self,path):
        # 将整个数据打包成一个大的batch
        files = list()
        for p in path:
            files_path = os.listdir(p)
            for f in files_path[:10] :
                files.append(os.path.join(p,f))
        batch_data = list()
        for i in tqdm(files,desc="load files" , unit='file'):
            with open(i , 'rb') as f:
                data_ = pickle.load(f)
            batch_data.append(data_)
        return batch_data
    def get_data(self , data , name):
        d = list()
        for i in range(len(data)):
            d.extend(data[i][name])
        return d
    def get_error(self):
        return self.error
    def deal_data(self,batch_datas):
        pre_audio = list()
        pre_visual = list()
        next_audio = list()
        next_visual = list()
        done = list()
        reward = list()
        action = list()
        self.error = list()
        for batch_id,batch_data in enumerate(tqdm(batch_datas,desc='deal batch_data' , unit="batch_datas")):
            data = batch_data[0]
            done_ = batch_data[2]
            frist_state = batch_data[3]
            if len(data) != 1:
                continue
            self.audio_ = self.get_data(data,'audio')
            self.tag_ = self.get_data(data,'rl_pred')
            self.visual_ = self.get_data(data,'camera')
            self.reward_ = self.get_data(data,'reward')
            self.done = list()
            self.next_audio = list()
            self.tag = list()
            self.next_visual = list()
            self.reward = list()
            for index,tag in enumerate(self.tag_):
                if tag == 3 or index == 199:
                    self.next_audio = self.audio_[:index+1]
                    self.next_visual = self.visual_[:index+1]
                    self.reward = self.reward_[:index+1]
                    self.tag = self.tag_[:index+1]
                    # 不包含stop信息，如需包含请index+1。具体就是指最后执行完stop之后不再会有相同的img产生
                    break
            self.next_audio.pop(0)
            self.next_visual.pop(0)
            self.reward.pop(0)
            self.tag.pop(0)
            for d in done_:
                if d[0] == False :
                    self.done.append(0)
                elif d[0] == True:
                    self.done.append(1)
            ##  get state , next_state
            self.pre_audio = [frist_state[0]['audio']] + self.next_audio[:-1]
            self.pre_visual = [frist_state[0]['camera']] + self.next_visual[:-1]
            if len(self.done) != len(self.tag):
                self.error.append(batch_id)
            pre_audio.extend(self.pre_audio)
            pre_visual.extend(self.pre_visual)
            next_audio.extend(self.next_audio)
            next_visual.extend(self.next_visual)
            done.extend(self.done)
            reward.extend(self.reward)
            action.extend(self.tag)
        return pre_audio,pre_visual, next_audio, next_visual,done,reward,action
    
class DiscreteSAC_CQL:
    def __init__(self, device, learning_rate=1e-4, alpha=0.1):
        self.device = device
        self.with_lagrange = False
        self.lr = learning_rate
        self.tau = 1e-2
        self.clip_grad_param = 1
        self.target_entropy = -4  # -dim(A)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        
        self.critic1 = Critic(128, 4, 128, 36).to(self.device)
        self.critic2 = Critic(128, 4, 128, 36).to(self.device)
        
        self.critic1_target = Critic(128, 4, 128, 36).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(128, 4, 128, 36).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        self.actor = Actor(128, 4, 128, 36).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
        self.softmax = nn.Softmax(dim=-1)

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
        entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
        policy_loss = -torch.mean(min_q * policy_dist) - self.alpha * entropy

        # actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        total_loss = q_loss + 0.5 * q_regularization + policy_loss
        return total_loss, q_loss, q_regularization, policy_loss

    def train_step(self, pre_audio, pre_visual, next_audio, next_visual, labels, reward, done):
        pre_audio, pre_visual, next_audio, next_visual, labels, reward, done = \
            pre_audio.float(), pre_visual.float(), next_audio.float(), next_visual.float(), \
            labels.long(), reward.float(), done.float()
        
        current_alpha = self.alpha
        Q_1 = self.critic1(pre_audio,pre_visual)
        Q_2 = self.critic2(pre_audio,pre_visual)
        min_Q = torch.min(Q_1,Q_2)
        action_probs, action_logits = self.actor(pre_audio, pre_visual)
        actor_loss = (action_probs * (current_alpha.to(self.device) * action_logits - min_Q )).sum(1).mean()
        log_pi = torch.sum(action_logits * action_probs, dim=1)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pi.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            action_probs, log_pis = self.actor(next_audio , next_visual)
            Q_target1_next = self.critic1_target(next_audio , next_visual)
            Q_target2_next = self.critic2_target(next_audio , next_visual)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = reward.unsqueeze(1) + (0.99 * (1 - done.unsqueeze(1)) * Q_target_next.sum(dim=1).unsqueeze(1))
            
        # Compute critic loss
        q1 = self.critic1(pre_audio , pre_visual)
        q2 = self.critic2(pre_audio , pre_visual)
        
        q1_ = q1.gather(1, labels.unsqueeze(1))
        q2_ = q2.gather(1, labels.unsqueeze(1))
        
        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)
        
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
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()
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

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql = DiscreteSAC_CQL(device)
    
    episode = 0
    # 数据集加载
    path = ['../data/RL/newdone']
    dataset = MyData(path=path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    num_epochs = 300
    for epoch in range(num_epochs):
        for batch_data in dataloader:
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_done, batch_reward, batch_labels = batch_data
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done = \
                batch_pre_audio.to(device), batch_pre_visual.to(device), batch_next_audio.to(device), batch_next_visual.to(device), \
                batch_labels.to(device), batch_reward.to(device), batch_done.to(device)

            actor_loss, alpha_loss, critic1_loss, critic2_loss, cql1_scaled_loss, cql2_scaled_loss, current_alpha, cql_alpha_loss, cql_alpha = cql.train_step(
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done
            )
            writer.add_scalar('Loss/actor_loss', actor_loss, episode)
            writer.add_scalar('Loss/alpha_loss', alpha_loss, episode)
            writer.add_scalar('Loss/critic1_loss', critic1_loss, episode)
            writer.add_scalar('Loss/critic2_loss', critic2_loss, episode)
            writer.add_scalar('Loss/cql1_scaled_loss', cql1_scaled_loss, episode)
            writer.add_scalar('Loss/cql2_scaled_loss', cql2_scaled_loss, episode)
            writer.add_scalar('Loss/current_alpha', current_alpha, episode)
            writer.add_scalar('Loss/cql_alpha_loss', cql_alpha_loss, episode)
            writer.add_scalar('Loss/cql_alpha', cql_alpha, episode)
            print(f'Epoch {epoch}: actor_loss {actor_loss}, critic1_loss {critic1_loss}, critic1_loss {critic2_loss}')
            episode+=1
    torch.save(cql.actor.state_dict(), './checkpoint/discrete_sac_actor.pth')
    torch.save(cql.critic1.state_dict(), './checkpoint/discrete_sac_critic1.pth')
    torch.save(cql.critic2.state_dict(), './checkpoint/discrete_sac_critic2.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/sac_cql_2'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/audio/avn_bel.log', level=logging.INFO)
    train()
