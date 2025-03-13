# 提取出audio的声音特征
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
from avf import AVFNet
class AVNet(nn.Module):
    def __init__(self,hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.model_path = '/data/Getuanhui/checkpoint/avnf_finnal.pth'
        self.avf = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.model_path))
        self.avf.eval()
        self.Q_net = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.out_put)
        )
        self.initialize_weights_uniform()
    def forward(self,audio,visual):
        target_q , combinencode = self.avf(audio , visual)
        action_q = self.Q_net(combinencode)
        return action_q , target_q
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
            for f in files_path :
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

class TQL:
    def __init__(self, model, device ,learning_rate=1e-5, alpha=1.0):
        self.lr = learning_rate
        self.gamma = 0.9
        self.model = model
        self.target_model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.alpha = alpha  # CQL's regularization coefficient
        self.criterion = nn.CrossEntropyLoss()
    def compute_loss(self, Q,target_Q , labels):
        # with torch.no_grad():
        #     reward = Q.sum()
        a_Q = Q.gather(1, labels.unsqueeze(1))
        max_Q = target_Q.max(1)[0].unsqueeze(1)
        # Q-value loss (mean squared error between predicted Q-values and target Q-values)
        q_loss = nn.MSELoss()(a_Q, max_Q)
        return  q_loss
    def train_step(self, pre_audio, pre_visual,next_audio,next_visual, labels, reward ,done,lr):
        self.lr = lr
        for param_grop in self.optimizer.param_groups:
            param_grop['lr'] = self.lr
        pre_audio = pre_audio.float()
        pre_visual = pre_visual.float()
        Q,target_Q = self.model(pre_audio, pre_visual)
        total_loss  = self.compute_loss(Q,target_Q ,labels)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
def train(logging):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVNet(128 , 4 ,width_dim=128 , height_dim=36).to(device)
    model.train()
    lr = 1e-5
    tql = TQL(model , device)
    path = [ '../data/RL/newdone']
    mydata = MyData(path=path)
    print(mydata.__len__())
    episode = 0
    numsepoch = 50
    dataloader = DataLoader(dataset=mydata , batch_size = 32 , shuffle = True)
    time_start = time.time()
    for epoch in range(numsepoch):
        # if epoch % 2 == 0:
        #     lr = lr / 10
        for batch_data in dataloader:
            batch_pre_audio, batch_pre_visual, batch_next_audio , batch_next_visual,batch_done, batch_reward ,batch_labels,= batch_data
            batch_pre_audio, batch_pre_visual, batch_next_audio , batch_next_visual, batch_labels, batch_reward ,batch_done= batch_pre_audio.to(device), batch_pre_visual.to(device), batch_next_audio.to(device) , batch_next_visual.to(device), batch_labels.to(device), batch_reward.to(device),batch_done.to(device)
            total_loss  = tql.train_step(batch_pre_audio, batch_pre_visual, batch_next_audio,batch_next_visual,batch_labels, batch_reward , batch_done,lr)
            print(f'epoch {epoch}, Episode {episode}, total_loss: {total_loss},lr {lr}')
            logging.info(f'epoch {epoch}, Episode {episode}, total_loss: {total_loss},lr {lr}')
            if episode % 10 == 0:
                    # Log total loss
                    writer.add_scalar('Loss/train', total_loss, episode)
            episode+=1
    logging.info(f"10 epoch cost time {(time.time() - time_start) % 60}m{(time.time() - time_start) // 60}s")
    torch.save(model.state_dict(), './checkpoint/TQL.pth')
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/TQL'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/audio/tql.log', level=logging.INFO)
    train(logging)