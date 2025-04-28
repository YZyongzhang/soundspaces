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
from acm.dsac.net.avf import AVFNet
# from net.avf import AVFNet

class LSTM(nn.Module):
    def __init__(self, input , hidden_dim , layer):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = './acm/checkpoint/avnf_finnal_1.pth'
        # self.model_path = '../checkpoint/avnf_finnal_1.pth'
        self.avf = AVFNet(hid_dim=128, out_put=4, width_dim=128, height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.model_path))
        self.avf.eval()
        self.input = input 
        self.out_dim = hidden_dim
        self.layer = layer
        self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.out_dim, num_layers=self.layer,batch_first=True).to(self.device)
        self.ht = torch.rand(self.layer*1,self.out_dim).cuda()
        self.ct = torch.rand(self.layer*1,self.out_dim).cuda()
    def forward(self,audio , visual , h0=None , c0=None):
        if h0 == None and c0 == None:
            audio_unsqueeze = audio.squeeze(1)
            visual_unsqueeze = visual.squeeze(1)
            combine_encode = self.avf(audio_unsqueeze , visual_unsqueeze)
            h0=torch.rand(self.layer*1,self.out_dim).cuda()
            c0=torch.rand(self.layer*1,self.out_dim).cuda()

            x, (ht,ct) = self.lstm(combine_encode,(h0,c0))
            return x
        else:
            audio_unsqueeze = audio.squeeze(1)
            visual_unsqueeze = visual.squeeze(1)
            combine_encode = self.avf(audio_unsqueeze , visual_unsqueeze)
            x, (ht,ct) = self.lstm(combine_encode,(h0,c0))
            return x , (ht,ct)
class Actor(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.policy_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.out_put)
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state):
        logits = self.policy_net(state)
        action_probs = self.softmax(logits)
        return action_probs
    
    def evaluate(self, state):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(self.device)
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
class Critic(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.out_put)
        )
    def forward(self, state):
        q1 = self.critic(state)
        return q1

class AVF(nn.Module):
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
    def forward(self,audio,visual):
        # print(visual.shape)
        visual = visual.permute(0,3,2,1)
        mel_features = torch.from_numpy(self.deal_audio(audio)).to(self.device)
        audio_fea = self.audio(mel_features)
        visual_fea = self.visual(visual)
        combinencode = self.mask(torch.cat((audio_fea , visual_fea) , 1))
        return combinencode
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