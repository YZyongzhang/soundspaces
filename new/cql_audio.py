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
# from new.audio_visual import visual , audio
from audio_visual import visual , audio
class Net(nn.Module):
    def __init__(self,hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
class MyData(Dataset):
    def __init__(self,data):
        self.audio_ = self.get_data(data,'audio')
        self.tag_ = self.get_data(data,'rl_pred')
        self.visual_ = self.get_data(data,'camera')
        self.reward_ = self.get_data(data,'reward')
        self.audio = list()
        self.tag = list()
        self.visual = list()
        self.reward = list()
        for index,tag in enumerate(self.tag_):
            if tag == 3:
                self.audio_ = self.audio_[:index]
                self.visual_ = self.visual_[:index]
                self.reward_ = self.reward_[:index]
                self.tag_ = self.tag_[:index]
                break
        # logging.info(self.tag_)
    def get_data(self , data , name):
        d = list()
        for i in range(len(data)):
            d.extend(data[i][name])
        return d
    def __len__(self):
        return len(self.audio_)
    def __getitem__(self,index):
        return self.audio_[index] ,self.visual_[index] ,self.tag_[index] ,self.reward_[index]
class CQL:
    def __init__(self, model, learning_rate=1e-5, alpha=1.0):
        self.lr = learning_rate
        self.gamma = 0.9
        self.model = model
        self.target_model = model
        self.device = model.device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.alpha = alpha  # CQL's regularization coefficient
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, Q,target_Q, audio , visual , labels):
        reward_ = torch.sum(Q)
        # Q-value loss (mean squared error between predicted Q-values and target Q-values)
        q_loss = nn.MSELoss()(Q, target_Q)
        
        # V-value loss (using the same target Q as the CQL algorithm)
        q_regularization = torch.mean(torch.square(Q - self.target_model(audio[1:]).gather(1, labels[:-1].unsqueeze(1))))
        
        

        # Total loss
        total_loss = q_loss + self.alpha * q_regularization
        return total_loss , q_loss , q_regularization ,reward_

    def train_step(self, audio, visual_input, labels, reward ,lr):
        self.lr = lr
        for param_grop in self.optimizer.param_groups:
            param_grop['lr'] = self.lr
        Q = self.model(audio[:-1]).gather(1, labels[:-1].unsqueeze(1))
        # gained best action Q value
        next_q_values = self.target_model(audio[1:])
        # next Q value 
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        # get the PI policy gianed action 
        target_q_values = reward[1:].unsqueeze(1) +   self.gamma * max_next_q_values
        # get the state value
        total_loss , q_loss , q_regularization,reward_ = self.compute_loss(Q,target_q_values ,audio , visual_input,labels)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), q_loss.item(),q_regularization.item(),reward_.item()
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def val(self, audio , visual , lable):
        with torch.no_grad():
            q = self.model(audio)
            val_loss = self.criterion(q , lable)
        return val_loss
    def deal_data(self,audio , visual):
        # audio [batch , 2 , 18000]
        # visual [batch , 128 , 128 ,4]
        # mel_audio []
        pass
        
def train(writer):
    path = "../data/RL/new_random"
    val_path = "../data/RL/random"
    files = os.listdir(path=path)
    val_files = os.listdir(path=val_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net().to(device)
    model = Net(64 , 4 ,width_dim=128 , height_dim=36).to(device)
    cql = CQL(model)
    num_episode = len(files)
    num_epoch = 3
    nums = 0
    for epoch in range(num_epoch):
        for episode in range(num_episode):
            lr = 0.01 * (0.1 ** (epoch // 2))
            model.train()
            file_path = os.path.join(path, files[episode])
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            dataset = MyData(data)
            dataloader = DataLoader(dataset=dataset, batch_size=dataset.__len__())
            
            for batch_idx, batch in enumerate(dataloader):
                batch_audio, batch_visual, batch_labels, batch_reward = batch
                batch_audio, batch_visual, batch_labels, batch_reward = batch_audio.to(device), batch_visual.to(device), batch_labels.to(device), batch_reward.to(device)

                total_loss, q_loss,q_regularization,reward  = cql.train_step(batch_audio, batch_visual, batch_labels, batch_reward ,lr)
                print(f'nums {nums} ,epoch {epoch}, Episode {episode}, total_loss: {total_loss}, q_loss {q_loss}, v_loss {q_regularization},lr {lr}')
                logging.info(f'epoch {epoch}, Episode {episode}, total_loss: {total_loss}, q_loss {q_loss}, v_loss {q_regularization} ,lr {lr}')
                if nums % 10 == 0:
                    val_file_path = os.path.join(val_path , val_files[episode])
                    with open(val_file_path , 'rb') as f:
                        val_data = pickle.load(f)
                    val_dataset = MyData(val_data)
                    dataloader = DataLoader(dataset=val_dataset, batch_size=dataset.__len__())
                    for batch_idx, batch in enumerate(dataloader):
                        batch_audio, batch_visual, batch_labels, batch_reward = batch
                        batch_audio, batch_visual, batch_labels, batch_reward = batch_audio.to(device), batch_visual.to(device), batch_labels.to(device), batch_reward.to(device)
                        val_loss = cql.val(batch_audio , batch_visual , batch_labels)
                        logging.info(f"val_loss {val_loss}")
                    # Log losses to TensorBoard
                    writer.add_scalar('loss/reward',reward ,nums)
                    writer.add_scalar('Loss/q_loss', q_loss, nums)
                    writer.add_scalar('Loss/q_regularization', q_regularization, nums)
                    # Log total loss
                    writer.add_scalar('Loss/train', total_loss, nums)
                    writer.add_scalar('loss/val' , val_loss , nums)
                    cql.update_target_network()
                nums+=1
    # torch.save(cql.model.state_dict() , 'rl_model_useolddata.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/cql_audio_2'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='output_cql_audio_2.log', level=logging.INFO)
    train(writer)