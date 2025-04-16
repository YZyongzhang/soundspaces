import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
import os , pdb
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
from dsac.data.angle_data import Data
# from dsac.data.uselmdb import Data
import time
class AVFNet(nn.Module):
    def __init__(self, hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.combin_dim = 3584
        self.audio = nn.Sequential(
            nn.Conv2d(2 , 32 , kernel_size = 5,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32 , 32, kernel_size = 5,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32 ,64 , kernel_size = 4,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64 ,64 , kernel_size = 3,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
        )
        self.visual = nn.Sequential(
            nn.Conv2d(4 , 32 , kernel_size = 5,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32 ,32 , kernel_size = 5,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32 ,64 , kernel_size = 4,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64 ,64 , kernel_size = 3,stride = 1,padding = 1),
            nn.MaxPool2d(2,2),
        )
        
        self.encode = nn.Sequential(
            nn.Linear(self.combin_dim,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU()
        )
        self.mask = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU()
        )
        self.action_net = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.out_put)
        )
    def forward(self,audio,visual):
        # pdb.set_trace()
        visual = visual.squeeze(0)
        audio = audio.squeeze(0)
        b , _ , _ ,_ = visual.shape
        visual = visual.permute(0,3,2,1)
        mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
        audio_fea = self.audio(mel_features)
        audio_fea = audio_fea.view(b , -1)
        visual_fea = self.visual(visual)
        visual_fea = visual_fea.view(b , -1)
        self.mask_input_dim = visual_fea.size(-1) + audio_fea.size(-1)
        combinencode = self.encode(torch.cat((audio_fea , visual_fea) , 1))
        combinfeature= self.mask(combinencode)
        action = self.action_net(combinfeature)
        return action
    #############################
    # 特征模型取中间层进行训练
    #############################
    # def forward(self,audio,visual):
    #     visual = visual.squeeze(0)
    #     audio = audio.squeeze(0)
    #     b , _ , _ ,_ = visual.shape
    #     visual = visual.permute(0,3,2,1)
    #     print(visual.shape)
    #     mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
    #     audio_fea = self.audio(mel_features)
    #     audio_fea = audio_fea.view(b , -1)
    #     visual_fea = self.visual(visual)
    #     visual_fea = visual_fea.view(b , -1)
    #     self.mask_input_dim = visual_fea.size(-1) + audio_fea.size(-1)
    #     print(self.mask_input_dim)
    #     combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
    #     return combinencode
    def deal_audio(self,audio):
        mel_features = []
        audio = audio.cpu().numpy()
        for i in range(audio.shape[0]):
            left_channel = audio[i, 0, :]
            right_channel = audio[i, 1, :]
            
            mel_left = librosa.feature.melspectrogram(y=left_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            mel_right = librosa.feature.melspectrogram(y=right_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            
            combined_mel = np.stack([mel_left, mel_right], axis=0)  # (2, 128, 时间帧数)
            mel_features.append(combined_mel)
        return np.array(mel_features)
    
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


def bi(batch_labels , model_action):
    batch_labels = batch_labels.squeeze(0).tolist()
    bi = 0
    for i in range(len(batch_labels)):
        if batch_labels[i] == model_action[i]:
            bi +=1 
    return bi/len(batch_labels)
    
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(device)
    
    # pretrain_path = '../data/checkpoint/acmcheckpoint/avf_muti_env_78350.pth'
    # model.load_state_dict(torch.load(pretrain_path))
    
    
    
    path = ['../data/RL/success_and_stop_data/level0' ,'../data/RL/success_and_stop_data/level1', '../data/RL/success_and_stop_data/level2']
    data_file = list()
    files0 = [os.path.join(path[0] , i) for i in os.listdir(path[0])]
    files1 = [os.path.join(path[1] , i) for i in os.listdir(path[1])]
    files2 = [os.path.join(path[2] , i) for i in os.listdir(path[2])]
    data_file.extend(files0)
    data_file.extend(files1)
    data_file.extend(files2)
    
    val_path = '../data/RL/valdata'
    val_files = [os.path.join(val_path  , i) for i in os.listdir(val_path)]
    # muti_env = '../data/RL/muti_env_data'
    # envs_data = [os.path.join(muti_env , i) for i in os.listdir(muti_env) if i != 'RLDATA.log']
    # for i in envs_data:
    #     data_file.extend(muti_data(i))
    begin_time = time.time()
    logging.info(f'now time {begin_time}')
    val_mydata = Data(path=val_files[100:200])
    val_dataloader = DataLoader(dataset=val_mydata , batch_size=1 , shuffle= True)
    # path = '../data/database/muti_env_data'
    mydata = Data(data_file)
    dataloader = DataLoader(dataset = mydata , batch_size = 1 , shuffle = True)
    numepoch = 500
    episode = 0
    lr = 1e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(numepoch):
        model.train()
        for batch_idx, batch_data in enumerate(dataloader):
            batch_pre_audio, batch_pre_visual, _ , _,_, _ ,batch_labels= batch_data
            optimizer.zero_grad()
            outputs = model(batch_pre_audio , batch_pre_visual)
            # pdb.set_trace()
            loss = criterion(outputs, batch_labels.squeeze(0))
            loss.backward()
            optimizer.step()
            print(f'epoch:{epoch} , eposide:{episode} , loss:{loss.item()}')
            writer.add_scalar('Loss/train', loss, episode)
            if episode % 200 == 0 :
                now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logging.info(f"now time :{now_time}")
            if episode % 5000 == 0:
                torch.save(model.state_dict() , f'../data/checkpoint/avf_checkpoint/avf_use_origin_data_{episode}.pth')
            episode +=1
        if epoch % 10 == 0 :
            model.eval()
            train_bi_list = list()
            val_bi_list = list()
            for batch_data in dataloader:
                batch_pre_audio, batch_pre_visual, batch_next_audio , batch_next_visual,batch_done, batch_reward ,batch_labels = batch_data
                # model_action = agent(batch_pre_audio,batch_pre_visual).max(-1)[1].tolist()
                model_action = model(batch_pre_audio.squeeze(0),batch_pre_visual.squeeze(0)).max(-1)[1].tolist()
                train_bi_list.append(bi(batch_labels=batch_labels , model_action=model_action))
            writer.add_scalar('val/train_max', max(train_bi_list), episode)
            writer.add_scalar('val/train_min', min(train_bi_list), episode)
            writer.add_scalar('val/train_mean', np.array(train_bi_list).mean(), episode)
            
            for batch_data in val_dataloader:
                batch_pre_audio, batch_pre_visual, batch_next_audio , batch_next_visual,batch_done, batch_reward ,batch_labels = batch_data
                # model_action = agent(batch_pre_audio,batch_pre_visual).max(-1)[1].tolist()
                model_action = model(batch_pre_audio.squeeze(0),batch_pre_visual.squeeze(0)).max(-1)[1].tolist()
                val_bi_list.append(bi(batch_labels=batch_labels , model_action=model_action))
            writer.add_scalar('val/val_max', max(val_bi_list), episode)
            writer.add_scalar('val/val_min', min(val_bi_list), episode)
            writer.add_scalar('val/val_mean', np.array(val_bi_list).mean(), episode)
        logging.info(f'use time {time.time() - begin_time}')
    torch.save(model.state_dict(), f'../data/checkpoint/avf_checkpoint/avf_use_origin_data_{episode}.pth')
    
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/avf_use_origin_data_train_and_val'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/avf_use_origin_data_train_and_val.log', level=logging.INFO)
    train()