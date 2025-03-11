import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
import os
import torch.optim as optim
import  pickle
import logging
class AudioNet(nn.Module):
    def __init__(self,hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.audio_net = nn.Sequential(
            nn.Conv2d(2 , 32 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Conv2d(32 ,64 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.width_dim*self.height_dim , 128),
            nn.ReLU(),
            nn.Linear(128 , self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim,32),
            nn.ReLU(),
            nn.Linear(32,self.out_put)
        )
        self.initialize_weights_uniform()
    def forward(self,audio):
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
        audio_feature = torch.from_numpy(np.array(mel_features)).to('cuda')
        return self.audio_net(audio_feature)
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
# 为了提取特征特意重写的model，这个是全部用于做转向任务的
# class AudioNet(nn.Module):
#     def __init__(self,hid_dim , out_put , width_dim,height_dim):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.out_put = out_put
#         self.width_dim = width_dim
#         self.height_dim = height_dim
#         self.audio = nn.Sequential(
#             nn.Conv2d(2 , 32 , kernel_size = 3,stride = 1,padding = 1),
#             nn.ReLU(),
#             nn.Conv2d(32 ,64 , kernel_size = 3,stride = 1,padding = 1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64*self.width_dim*self.height_dim , self.hid_dim),
#             nn.ReLU(),
#             nn.Linear(self.hid_dim , self.hid_dim),
#             nn.ReLU(),
#             nn.Linear(self.hid_dim , 64)
#         )
#         self.action = nn.Sequential(
#             nn.Linear(64,32),
#             nn.Linear(32,self.out_put)
#         )
#         self.initialize_weights_uniform()
#     def forward(self,audio):
#         mel_features = self.deal_audio(audio)
#         audio_feature = torch.from_numpy(np.array(mel_features)).to('cuda')
#         fea = self.audio(audio_feature)
#         return self.action(fea)
#     def deal_audio(self,audio):
#         mel_features = []
#         audio = audio.cpu().numpy()
#         # print(audio.shape)
#         for i in range(audio.shape[0]):
#             left_channel = audio[i, 0, :]
#             right_channel = audio[i, 1, :]
            
#             mel_left = librosa.feature.melspectrogram(y=left_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
#             mel_right = librosa.feature.melspectrogram(y=right_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            
#             combined_mel = np.stack([mel_left, mel_right], axis=0)  # (2, 128, 时间帧数)
#             mel_features.append(combined_mel)
#         return mel_features
#     def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 if hasattr(module, 'weight') and module.weight is not None:
#                     nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
#                 if hasattr(module, 'bias') and module.bias is not None:
#                     nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
class MyData(Dataset):
    def __init__(self,data):
        self.audio_ = data[0]['audio']
        self.tag_ = data[0]['rl_pred']
        self.audio = list()
        self.tag = list()
        for index,tag in enumerate(self.tag_):
            if tag !=3 and tag !=0:
                self.audio.append(self.audio_[index])
                self.tag.append(tag)
        logging.info(self.tag)
    def __len__(self):
        return len(self.audio)
    def __getitem__(self,index):
        return self.audio[index] , self.tag[index]
def train(logging):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioNet(64 , 4 ,width_dim=128 , height_dim=36).to(device)
    model.train()
    path = "../../data/new"
    files_dir = os.listdir(path)
    files_ = list()
    files = list()
    for file_dir in files_dir:
        files_.append(os.path.join(path,file_dir))
    for file_ in files_:
        for i in os.listdir(file_):
            files.append(os.path.join(file_ ,i))
    num_epochs = len(files)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(num_epochs):
        running_loss = 0.0
        run_id = 1
        logging.info(f'file:{files[epoch]}')
        with open(files[epoch], 'rb') as f:
            data = pickle.load(f)
        if  len(data) != 1:
            continue
        if 2 not in data[0]['rl_pred'] and 1 not in data[0]['rl_pred']:
            logging.info(f"not tag {data[0]['rl_pred']}")
            continue
        dataset = MyData(data)
        dataloader = DataLoader(dataset=dataset , batch_size=1 , shuffle=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue  # Skip the batch if it is None
            batch_audio,batch_tag = batch
            logging.info(f'labels:{batch_audio.shape}')
            batch_audio , batch_tag= batch_audio.to(device) , batch_tag.to(device)
            optimizer.zero_grad()
            outputs = model(batch_audio)

            loss = criterion(outputs, batch_tag)
            # print(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            run_id+=1

        # 在每个epoch结束时记录训练集的loss
        avg_train_loss = running_loss / run_id
        if avg_train_loss != 0 and epoch%50 == 0:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
    torch.save(model.state_dict(), 'audio_weights_release_1.pth')
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/auido_finnal'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/finnal.log', level=logging.INFO)
    train(logging)