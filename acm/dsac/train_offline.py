import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torch.distributions import Categorical 
from torch.nn.utils import clip_grad_norm_
import os , sys
from tqdm import tqdm
import random
import torch.optim as optim
import  pickle
import logging
import time
# from data.mydata import MyData
# from data.newdata import newMyData
from data.angle_data import Data
from net.sac_cql_1 import DiscreteSAC_CQL_1
def train_1():
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql = DiscreteSAC_CQL_1()
    episode = 0
    val_paths = ['../../data/RL/best_action_forward_angle_valdata']
    val_path = [os.path.join(val_paths[0] , i) for i in os.listdir(val_paths[0])]
    path_ = ['../../data/RL/muti_level_data/level0' , '../../data/RL/muti_level_data/level1', '../../data/RL/muti_level_data/level2']
    level0 = [os.path.join(path_[0] , f) for f in os.listdir(path_[0])]
    level1 = [os.path.join(path_[1] , f) for f in os.listdir(path_[1])]
    level2 = [os.path.join(path_[2] , f) for f in os.listdir(path_[2])]
    
    train_list = []
    train_list.extend(level0)
    train_dataset = Data(path=train_list)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    num_epochs = 500
    for epoch in range(num_epochs):
        if epoch == 50 :
            train_list.extend(level1)
            train_dataset = Data(path=train_list)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
        if epoch == 100 :
            train_dataset = Data(path=train_list)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
            train_list.extend(level2)

        for batch_data in train_dataloader:
            
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_done, batch_reward, batch_labels = batch_data
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done = \
                torch.stack(batch_pre_audio).to(device), torch.stack(batch_pre_visual).to(device), torch.stack(batch_next_audio).to(device), torch.stack(batch_next_visual).to(device), \
                torch.stack(batch_labels).to(device), torch.stack(batch_reward).to(device), torch.stack(batch_done).to(device)

            loss_dict= cql.train_step(
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done
            )
            
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value, episode)
            print(f'Epoch {epoch} , episode : {episode} , use time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s')
            # if episode % 100 == 0:
            #     reward , reward_dict = val(cql=cql  , val_path=val_path)
            #     for reward_name, reward_value in reward_dict.items():
            #         writer.add_scalar(f'Loss/{reward_name}', reward_value, episode)
            #         writer.add_scalar(f'Loss/{reward_name}_diff', reward_value - reward, episode)
            if episode % 1000 == 0:
                logging.info(f'Epoch {epoch} , episode : {episode} , use time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s')
            if episode % 10000 == 0 :
                torch.save(cql.state_dict(), f'./checkpoint/LSTM_1_{episode}.pth')
            episode+=1
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f'log : ./logs/loss/DSAC15 ; path : ./checkpoint/LSTM_1')
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    torch.save(cql.state_dict(), f'./checkpoint/LSTM_1.pth')
# def val(val_path , cql):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     file = [random.choice(val_path)]
#     val_data = Data(file)
#     pre_audio = torch.tensor(np.array(val_data.preaudio)).to(device)
#     pre_visual = torch.tensor(np.array(val_data.previsual)).to(device)
#     reward = torch.tensor(np.array(val_data.reward)).sum(-1)
#     reward_dict = cql.val_step(pre_audio , pre_visual)
#     return reward , reward_dict
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/DSAC16'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC16.log', level=logging.INFO,filemode='a')
    # train()
    train_1()# screen sac1