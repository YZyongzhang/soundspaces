import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from . import audio
# from audio import AudioNet
from torch.utils.data import Dataset , DataLoader 
import os
import torch.optim as optim
import  pickle
import logging
class VisualNet(nn.Module):
    def __init__(self,hid_dim , out_put , width_dim,height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model_path = "./new/checkpoint/audio_weights_fine_tune.pth"
        self.model_path = "./checkpoint/audio_weights_fine_tune.pth"
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim
        self.audio_net = audio.AudioNet(64 , 4 ,width_dim=128 , height_dim=36).to(self.device)
        self.audio_net.load_state_dict(torch.load(self.model_path))
        self.audio_net.eval()
        self.visual_net = nn.Sequential(
            nn.Conv2d(4 , 32 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Conv2d(32 ,64 , kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.width_dim*self.height_dim , 128)
        )
        self.f_net = nn.Sequential(
            nn.Linear(128 , self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,self.out_put)
        )
        self.initialize_weights_uniform()
    def forward(self,visual,audio):
        batch ,height , width , _ = visual.shape
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
        visual = visual*mask
        visual = visual.permute(0,3,2,1)
        x = self.visual_net(visual)
        return self.f_net(x)
    def initialize_weights_uniform(self, weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1)):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.uniform_(module.weight, a=weight_range[0], b=weight_range[1])
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.uniform_(module.bias, a=bias_range[0], b=bias_range[1])
class MyData(Dataset):
    def __init__(self,data):
        self.audio_ = data[0]['audio']
        self.tag_ = data[0]['rl_pred']
        self.visual_ = data[0]['camera']
        self.audio = list()
        self.tag = list()
        self.visual = list()
        num = 0
        for index,tag in enumerate(self.tag_):
            if tag !=3 and tag !=0:
                
                self.audio.append(self.audio_[index])
                self.visual.append(self.visual_[index])
                self.tag.append(tag)
            if tag == 0 and num <=4:
                    num +=1
                    self.audio.append(self.audio_[index])
                    self.visual.append(self.visual_[index])
                    self.tag.append(tag)
        self.audio.pop(0)
        self.tag.pop(0)
        self.visual.pop(0)
        logging.info(self.tag)
    def __len__(self):
        return len(self.audio)
    def __getitem__(self,index):
        return self.audio[index] ,self.visual[index] ,self.tag[index]
def get_data():
    
    path = "../../data/new"
    files_dir = os.listdir(path)
    files_ = list()
    files = list()
    for file_dir in files_dir:
        files_.append(os.path.join(path,file_dir))
    for file_ in files_:
        for i in os.listdir(file_):
            files.append(os.path.join(file_ ,i))
            
    path1 = '../../data/audio'
    f = os.listdir(path1)
    for w in f:
        files.append(os.path.join(path1 ,w))
    return files
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisualNet(64 , 4 ,width_dim=128 , height_dim=128).to(device)
    model.train()
    files = get_data()
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
        dataloader = DataLoader(dataset=dataset , batch_size=4 , shuffle=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue  # Skip the batch if it is None
            batch_audio,batch_visual,batch_tag = batch
            batch_audio , batch_visual , batch_tag= batch_audio.to(device) ,batch_visual.to(device) ,batch_tag.to(device)
            optimizer.zero_grad()
            outputs = model(batch_visual,batch_audio)

            loss = criterion(outputs, batch_tag)
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
        
    torch.save(model.state_dict(), 'visual_weights_3.pth')
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/visual4'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/visual.log', level=logging.INFO)
    train()