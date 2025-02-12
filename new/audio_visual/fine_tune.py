import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from audio import AudioNet
import os
import torch.optim as optim
import  pickle
import logging
model_path = "../checkpoint/audio_weights_release.pth"
import logging
class MyData(Dataset):
    def __init__(self,data):
        self.audio_ = data[0]['audio']
        self.tag_ = data[0]['rl_pred']
        self.audio = list()
        self.tag = list()
        num = 0
        for index,tag in enumerate(self.tag_):
            if tag !=3 and tag !=0:
                
                self.audio.append(self.audio_[index])
                self.tag.append(tag)
            if tag == 0 and num <=4:
                    num +=1
                    self.audio.append(self.audio_[index])
                    self.tag.append(tag)
        logging.info(self.tag)
    def __len__(self):
        return len(self.audio)
    def __getitem__(self,index):
        return self.audio[index] , self.tag[index]
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioNet(64 , 4 ,width_dim=128 , height_dim=36).to(device)
    model.load_state_dict(torch.load(model_path))
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
        dataloader = DataLoader(dataset=dataset , batch_size=4 , shuffle=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue  # Skip the batch if it is None
            batch_audio,batch_tag = batch
            batch_audio , batch_tag= batch_audio.to(device) , batch_tag.to(device)
            optimizer.zero_grad()
            outputs = model(batch_audio)

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
        
    torch.save(model.state_dict(), 'audio_weights_fine_tune.pth')
    writer.close()
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/auido_fine_tune'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/audio_fine_tune.log', level=logging.INFO)
    train()