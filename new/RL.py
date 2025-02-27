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
from audio_visual import visual , audio
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.action_dim = 4
        self.output_dim = 4
        self.x_dim = 36
        self.y_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualnet = visual.VisualNet(64 , 4 ,width_dim=128 , height_dim=128).to(self.device)
        v_model_path = "./audio_visual/visual_weights_1.pth"
        self.visualnet.load_state_dict(torch.load(v_model_path))
        self.visualnet.eval()
        a_model_path = "./checkpoint/audio_weights_fine_tune.pth"
        self.audio_net = audio.AudioNet(64 , 4 ,width_dim=128 , height_dim=36).to(self.device)
        self.audio_net.load_state_dict(torch.load(a_model_path))
        self.audio_net.eval()
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
        self.initialize_weights_uniform()
    def forward(self, audio, visual_input,labes):
        labes = labes.unsqueeze(1)
        batch ,height , width , _ = visual_input.shape
        result = self.audio_net(audio)
        x = torch.max(result,dim=1).indices
        mask = list()
        for i in x:
            mask_ = np.ones((height, width),dtype=np.uint8)
            height , width = mask_.shape
            if i.item() == 1:
                for h_i in range(height):
                    for w_i in range(width):
                    # left
                        if h_i + 2* w_i < 128:
                            mask_[h_i][w_i] = 0
            if i.item() == 2:
                for h_i in range(height):
                    for w_i in range(width):
                        # right
                        if h_i - 2* w_i < -128:
                            mask_[h_i][w_i] = 0
            if  i.item() == 0:
                   for h_i in range(height):
                    for w_i in range(width):
                        # mid
                        if h_i - 2* w_i > -128 and h_i + 2* w_i > 128:
                            mask_[h_i][w_i] = 0
            mask_ = mask_[:, :, np.newaxis]
            mask.append(mask_)
        mask = torch.from_numpy(np.array(mask)).to(self.device)
        red_overlay = np.array([255.0, 0.0, 0.0 , 1], dtype=np.float32)
        overlay = torch.from_numpy(red_overlay).to(self.device)
        # visual = overlay*(1-mask)*visual + mask*visual
        visual = visual_input*mask
        visual = visual.permute(0,3,2,1)
        combined_encode = self.visualnet.visual_net(visual)
        q_input_dim = torch.cat((combined_encode , labes) , dim=1)
        Q = self.q_net(q_input_dim)
        V = self.value_net(combined_encode)
        P = self.policy_net(combined_encode)
        return Q,V,P
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
        logging.info(self.tag_)
    def get_data(self , data , name):
        d = list()
        for i in range(len(data)):
            d.extend(data[i][name])
        return d
    def __len__(self):
        return len(self.audio_)
    def __getitem__(self,index):
        return self.audio_[index] ,self.visual_[index] ,self.tag_[index] ,self.reward_[index]            
class IQL:
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.device = model.device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def compute_loss(self, Q, V, P, target_Q, labels):
        target_Q = target_Q.unsqueeze(1)
        q_loss = F.mse_loss(Q, target_Q)
        v_loss = F.mse_loss(V, target_Q)
        log_probs = F.log_softmax(P, dim=1)
        action_loss = -torch.mean(torch.sum(log_probs * target_Q, dim=1))  
        total_loss = q_loss + v_loss + action_loss
        return total_loss , q_loss ,v_loss ,action_loss

    def train_step(self, audio, visual_input, labels, target_Q):
        Q, V, P = self.model(audio, visual_input, labels)
        total_loss , q_loss ,v_loss ,action_loss = self.compute_loss(Q, V, P, target_Q, labels)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item() ,q_loss.item(),v_loss.item() ,action_loss.item()
def train(writer):
    path = "../data/RL/random"
    files = os.listdir(path=path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    iql = IQL(model)
    num_epochs = len(files)
    s_q_loss , s_v_loss , s_action_loss , s_total_loss = (0, 0, 0, 0)
    for epoch in range(num_epochs):
        model.train()
        file_path = os.path.join(path, files[epoch])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        dataset = MyData(data)
        dataloader = DataLoader(dataset=dataset , batch_size=dataset.__len__())
        
        for batch_idx, batch in enumerate(dataloader):
            batch_audio, batch_visual, batch_labels  ,batch_reward = batch
            batch_audio, batch_visual, batch_labels  ,batch_reward= batch_audio.to(device), batch_visual.to(device), batch_labels.to(device) ,batch_reward.to(device)

            total_loss , q_loss ,v_loss , action_loss = iql.train_step(batch_audio,batch_visual, batch_labels, batch_reward)
            print(f'Episode {epoch}, total_loss: {total_loss} ,q_loss {q_loss} ,v_loss {v_loss} ,action_loss {action_loss }')
            if epoch % 1 == 0:
                # 记录每个损失到TensorBoard
                writer.add_scalar('Loss/q_loss', q_loss, epoch)
                writer.add_scalar('Loss/v_loss', v_loss,   epoch)
                writer.add_scalar('Loss/p_loss', action_loss, epoch)

                # 记录平均训练损失
                writer.add_scalar('Loss/train', total_loss, epoch)
                
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/rl_10'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='output_rl.log', level=logging.INFO)
    train(writer)