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
class AVFNet(nn.Module):
    def __init__(self,hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
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
            nn.Linear(self.hid_dim , 64),
        )
        self.visual = nn.Sequential(
            nn.Conv2d(4 , 32 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Conv2d(32 ,64 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*128*128 , 128),
            nn.ReLU(),
            nn.Linear(self.hid_dim , 64),
        )
        self.mask = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.action_net = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.out_put)
        )
        self.initialize_weights_uniform()
    def forward(self,audio,visual):
        # print(visual.shape)
        visual = visual.permute(0,3,2,1)
        mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
        audio_fea = self.audio(mel_features)
        visual_fea = self.visual(visual)
        combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
        action = self.action_net(combinencode)
        return action
    #############################
    # 特征模型取中间层进行训练
    #############################
    # def forward(self,audio,visual):
    #     # print(visual.shape)
    #     visual = visual.permute(0,3,2,1)
    #     mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
    #     audio_fea = self.audio(mel_features)
    #     visual_fea = self.visual(visual)
    #     combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
    #     return combinencode
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
        return np.array(mel_features)
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
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(device)
    model.train()
    path = "../data/RL/newdone"
    mydata = MyData(path=[path])
    dataloader = DataLoader(dataset = mydata , batch_size = 16 , shuffle = True)
    numepoch = 70
    lr = 1e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    episode = 0
    for epoch in range(numepoch):
        # if epoch % 5 == 0 and epoch != 0:
        #     for param_grop in optimizer.param_groups:
        #         param_grop['lr'] = lr / 10
        running_loss = 0.0
        run_id = 1
        for batch_idx, batch_data in enumerate(dataloader):
            batch_pre_audio, batch_pre_visual, _ , _,_, _ ,batch_labels= batch_data
            batch_pre_audio = batch_pre_audio.float().to(device)
            batch_pre_visual = batch_pre_visual.float().to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_pre_audio , batch_pre_visual)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            run_id+=1
            print(f'eposide:{episode} , loss:{loss.item()}')
            if episode % 10 == 0:
                avg_train_loss = running_loss / run_id
                writer.add_scalar('Loss/train', avg_train_loss, episode)
                run_id = 1
                running_loss = 0.0
            episode +=1      
    torch.save(model.state_dict(), './checkpoint/avnf_finnal.pth')
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/avf_finnal'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/audio/avf.log', level=logging.INFO)
    train()