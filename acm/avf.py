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
# from acm.dsac.data.angle_data import Data
from dsac.data.angle_data import Data
import time
class AVFNet(nn.Module):
    def __init__(self, hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.mask_input_dim = 0
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
        
        self.mask = nn.Sequential(
            nn.Linear(3584,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        self.action_net = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.out_put)
        )
        self.initialize_weights_uniform()
    def forward(self,audio,visual):
        visual = visual.squeeze(1)
        audio = audio.squeeze(1)
        b , _ , _ ,_ = visual.shape
        visual = visual.permute(0,3,2,1)
        mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
        audio_fea = self.audio(mel_features)
        audio_fea = audio_fea.view(b , -1)
        visual_fea = self.visual(visual)
        visual_fea = visual_fea.view(b , -1)
        self.mask_input_dim = visual_fea.size(-1) + audio_fea.size(-1)
        combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
        action = self.action_net(combinencode)
        return action
    #############################
    # 特征模型取中间层进行训练
    #############################
    # def forward(self,audio,visual):
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
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(device)
    model.load_state_dict(torch.load('../data/checkpoint/acmcheckpoint/avf_40000.pth'))
    model.train()
    path = "../data/RL/valdata"
    files = [os.path.join(path , i)  for i in os.listdir(path=path)]
    mydata = Data(files)
    dataloader = DataLoader(dataset = mydata , batch_size = 1 , shuffle = True)
    numepoch = 50
    lr = 1e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    episode = 0
    for epoch in range(numepoch):
        for batch_idx, batch_data in enumerate(dataloader):
            batch_pre_audio, batch_pre_visual, _ , _,_, _ ,batch_labels= batch_data
            batch_pre_audio = torch.stack(batch_pre_audio).to(device)
            batch_pre_visual = torch.stack(batch_pre_visual).to(device)
            batch_labels = torch.stack(batch_labels).to(device)
            optimizer.zero_grad()
            outputs = model(batch_pre_audio , batch_pre_visual)
            loss = criterion(outputs, batch_labels.squeeze(1))
            loss.backward()
            optimizer.step()
            print(f'eposide:{episode} , loss:{loss.item()}')
            writer.add_scalar('Loss/train', loss, episode)
            if episode % 200 == 0 :
                now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logging.info(f"now time :{now_time}")
            if episode % 2000 == 0:
                torch.save(model.state_dict() , f'../data/checkpoint/acmcheckpoint/avf_adddata_{episode}.pth')
            episode +=1
    torch.save(model.state_dict(), f'../data/checkpoint/acmcheckpoint/avf_adddata_{episode}.pth')
    
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/avf_adddata'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/avf_adddata.log', level=logging.INFO)
    train()