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
import time
# 自定义 AVFNet
import pdb
from data.use_combinencode_level import Data
from net.discrete_cql_sac import CQLSAC


class AVNet(nn.Module):
    def __init__(self, hid_dim, out_put, width_dim, height_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.out_put = out_put
        self.width_dim = width_dim
        self.height_dim = height_dim


        self.Q_net1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
        self.Q_net2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

    def forward(self, combinencode):
        q1 = self.Q_net1(combinencode)
        q2 = self.Q_net2(combinencode)
        logits = self.policy_net(combinencode)
        return q1, q2, logits
    

def train():
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVNet(128, 4, 128, 36).to(device)
    model.train()
    cql = CQLSAC(model, device)
    episode = 0
    
    
    # 数据集加载
    # database = '../../data/database/combinencodedatabase'
    database_path = ['../../data/database/combinencode_level/combinencode_0' , \
       '../../data/database/combinencode_level/combinencode_1' ,\
        '../../data/database/combinencode_level/combinencode_2']
    
    dataset = Data(database_path , limit_level = 0)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    num_epochs = 10000
    limit_level = 0
    for epoch in range(num_epochs):
        if epoch > 50 and epoch % 3 == 0 and limit_level <= 25:
            limit_level += 1
            dataset = Data(database_path , limit_level)
            dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
        
        if epoch > 200 and epoch % 3 == 0 and limit_level <= 50 and limit_level > 25:
            limit_level += 1
            dataset = Data(database_path , limit_level)
            dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
        # if limit_level < 50:
        #     limit_level += 1
        #     dataset = Data(database_path , limit_level)
        #     dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
        for batch_data in dataloader:
            # pdb.set_trace()
            batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch_data
            batch_done = batch_done.to(device)
                
            actor_loss, alpha_loss, critic1_loss, critic2_loss, cql1_scaled_loss, cql2_scaled_loss, current_alpha, cql_alpha_loss, cql_alpha= cql.learn(
                batch_pre_state ,batch_next_state, batch_labels, batch_reward, batch_done
            )
            writer.add_scalar('Loss/actor_loss', actor_loss, episode)
            writer.add_scalar('Loss/alpha_loss', alpha_loss, episode)
            writer.add_scalar('Loss/critic1_loss', critic1_loss, episode)
            writer.add_scalar('Loss/critic2_loss', critic2_loss, episode)
            writer.add_scalar('Loss/cql1_scaled_loss', cql1_scaled_loss, episode)
            writer.add_scalar('Loss/cql2_scaled_loss', cql2_scaled_loss, episode)
            writer.add_scalar('Loss/current_alpha', current_alpha, episode)
            writer.add_scalar('Loss/cql_alpha_loss', cql_alpha_loss, episode)
            writer.add_scalar('Loss/cql_alpha', cql_alpha, episode)
            print(f'Epoch {epoch} , episode : {episode}: actor_loss {actor_loss}, Q-Loss {(critic1_loss + critic2_loss)/2}, CQL-Reg {(cql1_scaled_loss + cql2_scaled_loss) /2}, cql_alpha{cql_alpha}')
            episode+=1
            
            # if episode % 1000 == 0 :
            #     logging.info(f"Epoch {epoch} , episode : {episode} ,time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
            # episode+=1
            # if episode % 10000 == 0 and episode != 0:
            #     torch.save(model.state_dict(), f'./checkpoint/DSAC_mutienv_cql_dn_combinencode_level_{episode}.pth')
                
                
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f'log : ./logs/loss/DSAC10 ; path : ./checkpoint/DSAC2.pth ; commit: batchsize = 32 ')
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    # torch.save(model.state_dict(), './checkpoint/DSAC_mutienv_cql_dn_combinencode_level.pth')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    time_stamp = "{0:%Y-%m-%d~%H:%M:%S/}".format(datetime.now())
    log_dir = './train/' + time_stamp
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC_use_cql_dn_combinencode_level.log', level=logging.INFO,filemode='a')
    train()
