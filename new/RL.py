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
from network import Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.action_dim = 4
        self.output_dim = 4
        self.x_dim = 36
        self.y_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
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
        visual_input = visual_input.permute(0, 3, 1, 2)
        audio_mask = self.audiomask(audio_input)
        audio_mask = audio_mask.view(audio_input.size(0), 2, self.y_dim, self.x_dim)
        masked_audio = audio_mask * audio_input
        
        audio_encode = self.audio_encoder(masked_audio)
        visual_encode = self.visual_encoder(visual_input)
        
        combined_encode = torch.cat((audio_encode, visual_encode), dim=1)
        combined_encode = self.fc1(combined_encode)
        # q_input_dim = torch.cat((combined_encode , labes) , dim=1)
        # Q = self.q_net(q_input_dim)
        # V = self.value_net(combined_encode)
        # P = self.policy_net(combined_encode)
        
        return combined_encode
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 初始化权重为均匀分布
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                # 初始化偏置为均匀分布
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
class MyData(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self ,idx):
        return self.data[idx]['audio'] , self.data[idx]['camera'] , self.data[idx]['rl_pred'] , self.data[idx]['reward']
class Train():
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma =0.99
        self.expectile = 0.5
        self.net = Net().to(device)
        self.state_net = Network().to(device)
        self.lr = 1e-3
        self.q_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        pass
    def rl_state(self,audio ,visual_input):
        audio  = audio.squeeze(0)
        visual_input  = visual_input.squeeze(0)
        state  = self.state_net.forward(audio , visual_input)
        next_state = torch.zeros_like(state)
        next_state[:-1] = state[1:]
        return state ,next_state
    def updata_q_net(self,state , next_state, rewards , label):
        label = label.squeeze(0)
        label = label.unsqueeze(1)
        rewards = rewards.squeeze(0)
        rewards = rewards.unsqueeze(1)
        with torch.no_grad():
            target_v = self.net.value_net(next_state)  # 目标价值网络预测的 V(s')
            q_target = rewards + self.gamma * target_v
        q_values = self.net.q_net(torch.cat([state, label], dim=1))
        loss_q = F.mse_loss(q_values, q_target)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        return loss_q.item()
    def updata_p_net(self , state , action):
        """
        更新策略网络
        """
        action = action.squeeze(0)
        action = action.unsqueeze(1)
        with torch.no_grad():
            q_values = self.net.q_net(torch.cat([state, action], dim=1))
            v_values = self.net.value_net(state)
            advantages = q_values - v_values  # 计算优势函数

        # 策略网络的损失，最大化优势函数
        log_probs = -self.net.policy_net(state)  # 假设策略输出为概率分布的对数值
        loss_policy = -(log_probs * advantages).mean()

        # 更新策略网络
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.policy_optimizer.step()

        return loss_policy.item()
    def update_v_net(self,state,action):
        action = action.squeeze(0)
        action = action.unsqueeze(1)
        with torch.no_grad():
            q_values = self.net.q_net(torch.cat([state, action], dim=1))

        # 计算 expectile regression 损失
        v_values = self.net.value_net(state)
        diff = q_values - v_values
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)  # 使用 expectile 权重
        loss_v = (weight * (diff ** 2)).mean()

        # 更新价值网络
        self.value_optimizer.zero_grad()
        loss_v.backward()
        self.value_optimizer.step()

        return loss_v.item()
def begin():
    path = "../data/audio"
    files = os.listdir(path=path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    train = Train()
    num_epochs = 800
    for epoch in range(num_epochs):
        train.net.train()
        file_path = os.path.join(path, files[epoch])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        dataset = MyData(data)
        dataloader = DataLoader(dataset=dataset , batch_size=1 , shuffle=True )
        
        for batch_idx, batch in enumerate(dataloader):
            batch_audio, batch_visual, batch_labels  ,batch_reward = batch
            batch_audio, batch_visual, batch_labels  ,batch_reward= batch_audio.to(device), batch_visual.to(device), batch_labels.to(device) ,batch_reward.to(device)
            state , next_state = train.rl_state(batch_audio , batch_visual)
            q_loss = train.updata_q_net(state.detach() , next_state.detach(),batch_reward , batch_labels.detach())
            v_loss = train.update_v_net(state.detach() , batch_labels.detach())
            p_loss = train.updata_p_net(state.detach(), batch_labels.detach())
            # 计算平均训练损失
            avg_train_loss = (q_loss + v_loss + p_loss) / 3

            # 记录每个损失到TensorBoard
            writer.add_scalar('Loss/q_loss', q_loss, epoch)
            writer.add_scalar('Loss/v_loss', v_loss, epoch)
            writer.add_scalar('Loss/p_loss', p_loss, epoch)

            # 记录平均训练损失
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f} , q_loss:{q_loss:.4f}, v_loss:{v_loss:.4f},p_loss:{p_loss:.4f}")
        # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f} , q_loss:{q_loss:.4f}, v_loss:{v_loss:.4f},p_loss:{p_loss:.4f}")
    torch.save(train.net.state_dict(), 'model_weights_rl.pth')
    writer.close()  # 关闭TensorBoard的SummaryWriter
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/rl'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='output_rl.log', level=logging.INFO)
    begin()