import torch
import pickle
import logging
import os , re
import numpy as np
import librosa
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.x_dim = 36
        self.y_dim = 128
        self.output_dim = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.audiomask = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.x_dim * self.y_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * self.x_dim * self.y_dim),
            nn.Sigmoid()
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 9, self.hidden_dim)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, self.hidden_dim)
        )
        
        self.fc1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.initialize_weights_uniform()
    
    def forward(self, audio, visual_input):
        mel_features = []
        # audio = np.array(audio)
        audio = audio.cpu().numpy()
        for i in range(audio.shape[0]):
            left_channel = audio[i, 0, :]
            right_channel = audio[i, 1, :]
            
            mel_left = librosa.feature.melspectrogram(y=left_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            mel_right = librosa.feature.melspectrogram(y=right_channel, sr=18000, n_fft=1024, hop_length=512, n_mels=128)
            
            combined_mel = np.stack([mel_left, mel_right], axis=0)  # (2, 128, 时间帧数)
            mel_features.append(combined_mel)

        audio_input= np.array(mel_features)
        audio_input = torch.from_numpy(audio_input).to(self.device)
        # visual_input = torch.from_numpy(visual_input)
        visual_input = visual_input.permute(0, 3, 1, 2)
        audio_mask = self.audiomask(audio_input)
        audio_mask = audio_mask.view(audio_input.size(0), 2, self.y_dim, self.x_dim)
        masked_audio = audio_mask * audio_input
        
        audio_encode = self.audio_encoder(masked_audio)
        visual_encode = self.visual_encoder(visual_input)
        
        combined_encode = torch.cat((audio_encode, visual_encode), dim=1)
        combined_encode = self.fc1(combined_encode)
        av_out = self.fc2(combined_encode)
        
        return av_out
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        """
        初始化网络的权重和偏置为均匀分布。
        
        Args:
            weight_range (tuple): 权重的均匀分布范围 (a, b)。
            bias_range (tuple): 偏置的均匀分布范围 (a, b)。
        """
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 初始化权重为均匀分布
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                # 初始化偏置为均匀分布
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
class MyData(Dataset):
    def __init__(self, data):
        self.data = data
        if np.where(self.data[0]['rl_pred'] == 3)[0].size == 0:
            self.index_3 = 61
        else:
            self.index_3 = np.where(self.data[0]['rl_pred'] == 3)[0][0].item()
        self.audio = list()
        self.visual = list()
        self.pred_tag = list()
        for data in self.data:
            self.audio.append(data['audio'][:self.index_3])
            self.visual.append(data['camera'][:self.index_3])
            self.pred_tag.append(data['rl_pred'][:self.index_3])

        self.audio = torch.from_numpy(np.array(self.audio))
        self.visual = torch.from_numpy(np.array(self.visual))
        self.pred_tag = torch.from_numpy(np.array(self.pred_tag))
        logging.info(f'audio:{self.audio.shape}')
        logging.info(f'v:{self.visual.shape}')
        logging.info(f'p:{self.pred_tag.shape}')
        # # Flatten the data so that each sample is an individual entry
        self.audio = self.audio.view(-1, *self.audio.shape[2:])
        self.visual = self.visual.view(-1, *self.visual.shape[2:])
        self.pred_tag = self.pred_tag.view(-1, *self.pred_tag.shape[2:])
        logging.info(f'index:{self.index_3}')
        logging.info(f'audio:{self.audio.shape}')
        logging.info(f'v:{self.visual.shape}')
        logging.info(f'p:{self.pred_tag.shape}')
    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        # Return a sample from audio, visual and pred_tag
        
        return self.audio[idx], self.visual[idx], self.pred_tag[idx]
def begin():
    path = "../data/new"
    files_dir = os.listdir(path)
    files_ = list()
    files = list()
    for file_dir in files_dir:
        files_.append(os.path.join(path,file_dir))
    for file_ in files_:
        for i in os.listdir(file_):
            files.append(os.path.join(file_ ,i))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = Network().to(device)
    
    
    num_epochs = len(files)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # # 将文件分为训练集和验证集，假设验证集占数据的50%
    # num_files = len(files)
    # val_split = int(num_files / 2)
    # train_files = files[:val_split]
    # val_files = files[val_split:]

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        run_id = 1
        # 训练集
        logging.info(f'file:{files[epoch]}')
        with open(files[epoch], 'rb') as f:
            data = pickle.load(f)
        if  len(data) != 1:
            continue
        dataset = MyData(data)
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataloader = DataLoader(dataset=dataset , batch_size=4 , shuffle=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue  # Skip the batch if it is None
            batch_audio, batch_visual, batch_labels = batch
            logging.info(f'labels:{batch_labels}')
            batch_audio, batch_visual, batch_labels = batch_audio.to(device), batch_visual.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            # batch_audio  = torch.zeros([16, 2, 18000])
            # batch_visual = torch.zeros([16, 128, 128, 4]).to(device)
            outputs = model(batch_audio, batch_visual)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            run_id+=1

        # 在每个epoch结束时记录训练集的loss
        avg_train_loss = running_loss / run_id
        if avg_train_loss != 0 and epoch%50 == 0:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # # 验证集测试部分
        # model.eval()  # 设置为评估模式，关闭dropout等
        # val_loss = 0.0
        # val_id = 1
        # with torch.no_grad():  # 不计算梯度，节省内存 
        #     file_path = os.path.join(path, val_files[epoch])
        #     with open(file_path, 'rb') as f:
        #         data = pickle.load(f)
        #     dataset = MyData(data)
        #     # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        #     dataloader = DataLoader(dataset=dataset , batch_size=32 , shuffle=True , collate_fn=dataset.collate_fn)

        #     for batch_idx, batch in enumerate(dataloader):
        #         if batch is None:
        #             continue  # Skip the batch if it is None
        #         batch_audio, batch_visual, batch_labels = batch
        #         logging.info(f'labels:{batch_labels}')
        #         batch_audio, batch_visual, batch_labels = batch_audio.to(device), batch_visual.to(device), batch_labels.to(device)
        #         # batch_audio  = torch.zeros([16, 2, 18000])
        #         # batch_visual = torch.zeros([16, 128, 128, 4]).to(device)
        #         outputs = model(batch_audio, batch_visual)
        #         loss = criterion(outputs, batch_labels)
        #         val_loss += loss.item()
        #         val_id+=1

        # avg_val_loss = val_loss / val_id
        # # avg_val_loss = val_loss
        # if avg_val_loss != 0:
        #     writer.add_scalar('Loss/val', avg_val_loss, epoch)

        # logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    # torch.save(model.state_dict(), 'model_weights_release.pth')
    writer.close()  # 关闭TensorBoard的SummaryWriter
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/new_data'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/output.log', level=logging.INFO)
    begin()