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
class AudioNet(nn.Module):
    def __init__(self,hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.audio = nn.Sequential(
            nn.Conv2d(2 , 32 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Conv2d(32 ,64 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.width_dim*self.height_dim , self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim , self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim , 64)
        )
        self.action = nn.Sequential(
            nn.Linear(64,32),
            nn.Linear(32,self.out_put)
        )
        self.initialize_weights_uniform()
    def forward(self,audio):
        mel_features = self.deal_audio(audio)
        audio_feature = torch.from_numpy(np.array(mel_features)).to('cuda')
        fea = self.audio(audio_feature)
        return self.action(fea)
    def deal_audio(self,audio):
        mel_features = []
        audio = audio.cpu().numpy()
        # print(audio.shape)
        for i in range(audio.shape[0]):
            left_channel = audio[i, 0, :]
            right_channel = audio[i, 1, :]
            
            mel_left = librosa.feature.melspectrogram(y=left_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            mel_right = librosa.feature.melspectrogram(y=right_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            
            combined_mel = np.stack([mel_left, mel_right], axis=0)  # (2, 128, 时间帧数)
            mel_features.append(combined_mel)
        return mel_features
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
        _,_,self.audio ,_,_,_, self.action  = self.deal_data(batch_data)
    def __len__(self):
        return len(self.audio)
    def __getitem__(self,index):
        return self.audio[index] , self.action[index]
    def load_data(self,path):
        # 将整个数据打包成一个大的batch
        files = list()
        files_path = os.listdir(path)
        for f in files_path :
            files.append(os.path.join(path,f))
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
    def deal_data(self,batch_datas):
        pre_audio = list()
        pre_visual = list()
        next_audio = list()
        next_visual = list()
        done = list()
        reward = list()
        action = list()
        
        for batch_data  in tqdm(batch_datas,desc='deal batch_data' , unit="batch_datas"):
            data = batch_data[0]
            done_ = batch_data[2]
            frist_state = batch_data[3]
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
                if tag == 3:
                    self.next_audio = self.audio_[:index]
                    self.next_visual = self.visual_[:index]
                    self.reward = self.reward_[:index]
                    self.tag = self.tag_[:index]
                    # 不包含stop信息，如需包含请index+1
                    break
            self.next_audio.pop(0)
            self.next_visual.pop(0)
            self.tag.pop(0)
            for d in done_:
                if d[0] == False :
                    self.done.append(0)
                elif d[0] == True:
                    self.done.append(1)
            ##  get state , next_state
            self.pre_audio = [frist_state[0]['audio']] + self.next_audio[:-1]
            self.pre_visual = [frist_state[0]['camera']] + self.next_visual[:-1]
            
            pre_audio.extend(self.pre_audio)
            pre_visual.extend(self.pre_audio)
            next_audio.extend(self.next_audio)
            next_visual.extend(self.next_visual)
            done.extend(self.done)
            reward.extend(self.reward)
            action.extend(self.tag)
        return pre_audio,pre_visual, next_audio, next_visual,done,reward,action
        
def train(logging):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioNet(128 , 4 ,width_dim=128 , height_dim=36).to(device)
    model.train()
    lr = 1e-5
    path = "../data/RL/newdone"
    mydata = MyData(path=path)
    num_episode = mydata.__len__()
    episode = 0
    numsepoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset=mydata , batch_size = 32 , shuffle = True)
    time_start = time.time()
    for epoch in range(numsepoch):
        for batch_data in dataloader:
            audio , action = batch_data
            audio , action = audio.to(device) , action.to(device)
            outputs = model(audio)
            loss = criterion(outputs , action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train',loss , episode)
        
            logging.info(f"Epoch {epoch},episode {episode}, Train Loss: {loss:.4f} lr = {lr} ")
            print(f"Epoch {epoch},episode {episode}, Train Loss: {loss:.4f} lr = {lr}")
            if episode % 5000 == 0 and episode != 0:
                for param_grop in optimizer.param_groups:
                    param_grop['lr'] = lr / 10
                    lr = param_grop['lr']
            episode+=1
    logging.info(f"10 epoch cost time {(time.time() - time_start) % 60}m{(time.time() - time_start) // 60}s")
    torch.save(model.state_dict(), 'audio_weights_2.pth')
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/auido3'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/audio/audio1.log', level=logging.INFO)
    train(logging)