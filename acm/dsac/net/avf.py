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
import pdb
import time
class AVFNet(nn.Module):
    def __init__(self, hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.mask_dim = 3584
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
        
        # self.mask = nn.Sequential(
        #     nn.Linear(3584,512),
        #     nn.ReLU(),
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Linear(256,256),
        #     nn.ReLU(),
        #     nn.Linear(256,128),
        #     nn.ReLU(),
        #     nn.Linear(128,128),
        # )
        self.mask = nn.Sequential(
            nn.Linear(self.mask_dim,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.out_put)
        )
    # def forward(self,audio,visual):
    #     # pdb.set_trace()
    #     visual = visual.squeeze(0)
    #     audio = audio.squeeze(0)
    #     b , _ , _ ,_ = visual.shape
    #     visual = visual.permute(0,3,2,1)
    #     mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
    #     audio_fea = self.audio(mel_features)
    #     audio_fea = audio_fea.view(b , -1)
    #     visual_fea = self.visual(visual)
    #     visual_fea = visual_fea.view(b , -1)
    #     self.mask_input_dim = visual_fea.size(-1) + audio_fea.size(-1)
    #     combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
    #     action = self.action_net(combinencode)
    #     return action
    #############################
    # 特征模型取中间层进行训练
    #############################
    def forward(self,audio,visual):
        b , _ , _ ,_ = visual.shape
        visual = visual.permute(0,3,2,1)
        mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
        audio_fea = self.audio(mel_features)
        audio_fea = audio_fea.view(b , -1)
        visual_fea = self.visual(visual)
        visual_fea = visual_fea.view(b , -1)
        self.mask_input_dim = visual_fea.size(-1) + audio_fea.size(-1)
        combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
        return combinencode
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