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
import torch.optim as optim
import  pickle
import logging
import time
# from data.mydata import MyData
# from data.newdata import newMyData
from data.data1 import Data
from net.sac_cql import DiscreteSAC_CQL
from net.sac_cql_1 import DiscreteSAC_CQL_1
def train():
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql = DiscreteSAC_CQL()
    episode = 0
    # 数据集加载
    path = ['../../data/RL/deal_data']
    dataset = Data(path=path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_data in dataloader:
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_done, batch_reward, batch_labels = batch_data
            batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done = \
                batch_pre_audio.to(device), batch_pre_visual.to(device), batch_next_audio.to(device), batch_next_visual.to(device), \
                batch_labels.to(device), batch_reward.to(device), batch_done.to(device)

            policy_loss, alpha_loss, critic1_loss, critic2_loss, cql1_scaled_loss, cql2_scaled_loss, current_alpha, cql_alpha_loss, cql_alpha= cql.train_step(
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done
            )
            writer.add_scalar('Loss/critic1_loss', critic1_loss, episode)
            writer.add_scalar('Loss/critic2_loss', critic2_loss, episode)
            writer.add_scalar('Loss/cql1_scaled_loss', cql1_scaled_loss, episode)
            writer.add_scalar('Loss/cql2_scaled_loss', cql2_scaled_loss, episode)
            writer.add_scalar('Loss/policy_loss', policy_loss, episode)
            writer.add_scalar('Loss/alpha_loss', alpha_loss, episode)
            writer.add_scalar('Loss/alpha', current_alpha, episode)
            writer.add_scalar('Loss/cql_alpha_loss', cql_alpha_loss, episode)
            writer.add_scalar('Loss/cql_alpha', cql_alpha, episode)
            print(f'Epoch {epoch} , episode : {episode}: policy Loss {policy_loss}, Q-1 {critic1_loss}, Q-2 {critic2_loss}')
            episode+=1
            if episode % 10000 == 0 and episode != 0:
                torch.save(cql.policy.state_dict(), f'./checkpoint/policy_1_{episode}.pth')
                torch.save(cql.critic1.state_dict(), f'./checkpoint/critic1_1_{episode}.pth')
                torch.save(cql.critic2.state_dict(), f'./checkpoint/critic2_1_{episode}.pth')
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    logging.info(f"begin time : {current_time_str} now time :{now_time}")
    logging.info(f'log : ./logs/loss/DSAC5 ; path : ./checkpoint/policy1.pth  ./checkpoint/critic11.pth  ./checkpoint/critic21.pth; commit: alpha = current ')
    logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    torch.save(cql.policy.state_dict(), './checkpoint/policy_1.pth')
    torch.save(cql.critic1.state_dict(), './checkpoint/critic1_1.pth')
    torch.save(cql.critic2.state_dict(), './checkpoint/critic2_1.pth')
def train_1():
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql = DiscreteSAC_CQL_1()
    episode = 0

    path = ['../../data/RL/mydata']
    paths = os.listdir(path=path[0])
    path_files = list()
    for i in paths:
        path_files.append(os.path.join(path[0] , i))
    num_split = 50
    split_lists = np.array_split(path_files, num_split)
    split_lists = [list(sublist) for sublist in split_lists]

    num_epochs = 500
    for epoch in range(num_epochs):
        
        for i in split_lists:
            dataset = Data(path=i)
            dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
            for batch_data in dataloader:
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_done, batch_reward, batch_labels = batch_data
                batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done = \
                    batch_pre_audio.to(device), batch_pre_visual.to(device), batch_next_audio.to(device), batch_next_visual.to(device), \
                    batch_labels.to(device), batch_reward.to(device), batch_done.to(device)

                loss_dict= cql.train_step(
                    batch_pre_audio, batch_pre_visual, batch_next_audio, batch_next_visual, batch_labels, batch_reward, batch_done
                )
                
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'Loss/{loss_name}', loss_value, episode)
                print(f'Epoch {epoch} , episode : {episode} , use time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s')
                if episode % 1000 == 0:
                    logging.info(f'Epoch {epoch} , episode : {episode} , use time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s')
                episode+=1
                if episode % 5000 == 0 and episode != 0:
                    torch.save(cql.policy.state_dict(), f'./checkpoint/policy_1_{episode}.pth')
                    torch.save(cql.critic1.state_dict(), f'./checkpoint/critic1_1_{episode}.pth')
                    torch.save(cql.critic2.state_dict(), f'./checkpoint/critic2_1_{episode}.pth')
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    logging.info(f"begin time : {current_time_str} now time :{now_time}")
    logging.info(f'log : ./logs/loss/DSAC6 ; path : ./checkpoint/policy_1.pth  ./checkpoint/critic1_1.pth  ./checkpoint/critic2_1.pth; ')
    logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
    torch.save(cql.policy.state_dict(), './checkpoint/policy_1.pth')
    torch.save(cql.critic1.state_dict(), './checkpoint/critic1_1.pth')
    torch.save(cql.critic2.state_dict(), './checkpoint/critic2_1.pth')
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/loss/DSAC9'
    writer = SummaryWriter(log_dir)
    logging.basicConfig(filename='./logs/DSAC9.log', level=logging.INFO,filemode='a')
    # train()
    train_1()# screen sac1