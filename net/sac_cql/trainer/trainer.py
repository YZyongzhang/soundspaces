import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torch.utils.data._utils.collate import default_collate
import os , math
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
import time
import pdb
import sys
import ray
sys.path.append('/home/getuanhui/project/sound-spaces')
from yz.config import agent_config , config

from yz.net import use_combinencode_level_data as Data
from yz.net import use_combinencode_data as Val_Data
from yz.net.utils import lmdb_sampler
from yz.net.sac_cql.SAC_CQL import DiscreteSAC_CQL as SAC_CQL
from yz.net.sac_cql.SAC_CQL import Critic_Actor



def Train(ckpt_dir,writer):
    database_dir = agent_config.RELATIVE_DATABASE_DIR
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Critic_Actor(action_dim=4).to(device)
    model.train()
    target_model = Critic_Actor(action_dim=4).to(device)
    target_model.train()
    cql = SAC_CQL(model, target_model , device)
    episode = 0
    
    database_path = [os.path.join(f'{database_dir}/new_key_shffule_mutienv_data_combinencode_level/' , i) \
        for i in os.listdir(f'{database_dir}/new_key_shffule_mutienv_data_combinencode_level/')]
    init_sample_dict = {
        'level_0':1,
        'level_1':0,
        'level_2':0,
    }
    sampler = lmdb_sampler(database_path, shuffle=True)
    sampler.sample_data(sample=init_sample_dict)
    dataset = Data(database_path)
    dataloader = DataLoader(dataset=dataset,\
        sampler=sampler ,batch_size=256)
    
    val_data_base = f'{database_dir}/val_database_combinencode/'
    val_dataloader = DataLoader(dataset=Val_Data(val_data_base) , batch_size=256)
    
    num_epochs = 500

    for epoch in range(num_epochs):
        if epoch > 50 and epoch % 10 == 0 and epoch < 200:
            if 0.1 * ( (epoch - 50) / 10 ) <= 1:
                level_1_rate = 0.1 * ( (epoch - 50) / 10 )
            else:
                level_1_rate = 1
            sample_dict = {
                'level_0':1,
                'level_1':level_1_rate,
                'level_2':0,
            }
            sampler.sample_data(sample=sample_dict)
            dataloader = DataLoader(dataset=dataset,\
                    sampler=sampler ,batch_size=256)
        
        if epoch > 200 and epoch % 10 == 0 and epoch < 400:
            if 0.1 * ( (epoch - 200) / 10 ) <= 1:
                level_2_rate = 0.1 * ( (epoch - 200) / 10 )
            else:
                level_2_rate = 1
            sample_dict = {
                'level_0':1,
                'level_1':1,
                'level_2':level_2_rate,
            }
            sampler.sample_data(sample=sample_dict)
            dataloader = DataLoader(dataset=dataset,\
                    sampler=sampler ,batch_size=256)
        for batch_data in dataloader:
            batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch_data
            batch_done = batch_done.to(device)
                
            loss_dict = cql.train_step(
                batch_pre_state ,batch_next_state, batch_labels, batch_reward, batch_done
            )
            print(f'epoch:{epoch} , episode {episode}' + ",".join([f"{k}: {v}" for k, v in loss_dict.items()]))
            for name, item in loss_dict.items():
                writer.add_scalar(f'loss/{name}', item, episode)
            episode += 1
            if epoch % 25 == 0 and episode != 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_{episode}_{epoch}.pth')
            
        if epoch % 2 == 0:
            model.eval()
            total_q_1_value = 0
            total_q_2_value = 0
            total_double_q_min_value = 0
            train_total_q_1_value = 0
            train_total_q_2_value = 0
            train_total_double_q_min_value=0
            q_1_accuracy = 0
            q_2_accuracy = 0
            double_q_min_accuracy = 0
            actor_accuracy = 0
            train_q_1_accuracy = 0
            train_q_2_accuracy = 0
            train_double_q_min_accuracy=0
            train_actor_accuracy = 0
            num_batches = 0
            with torch.no_grad():
                for val_batch in val_dataloader: 
                    batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = val_batch

                    # 得到 Q 值 (或者策略分布)
                    q_1 , q_2 , action = model(batch_pre_state)  # [batch, num_actions]
                    q_1_action = torch.argmax(q_1, dim=1)  # greedy action
                    q_2_action = torch.argmax(q_2, dim=1)
                    double_q_min_action = torch.argmax(torch.min(q_1,q_2),dim = 1)
                    actor_action = torch.argmax(action, dim=1)
                    q_1_selected = q_1.gather(1, q_1_action.unsqueeze(1)).squeeze(1)
                    q_2_selected = q_2.gather(1, q_2_action.unsqueeze(1)).squeeze(1)
                    double_q_min_action_selected = torch.min(q_1,q_2).gather(1 , double_q_min_action.unsqueeze(1)).squeeze(1)
                    total_q_1_value += q_1_selected.mean().item()
                    total_q_2_value += q_2_selected.mean().item()
                    total_double_q_min_value += double_q_min_action_selected.mean().item()
                    q_1_accuracy += ((q_1_action == batch_labels).sum().item())/ batch_labels.size(0)
                    q_2_accuracy += ((q_2_action == batch_labels).sum().item())/ batch_labels.size(0)
                    double_q_min_accuracy += ((double_q_min_action == batch_labels).sum().item())/ batch_labels.size(0)
                    actor_accuracy += ((actor_action == batch_labels).sum().item())/ batch_labels.size(0)
                    num_batches += 1
                    if num_batches >= 10:  # 只验证10个 batch 就够了，别太频繁
                        break
                writer.add_scalar('val/total_q_1_value', total_q_1_value / 10, epoch)
                writer.add_scalar('val/total_q_2_value', total_q_2_value / 10, epoch)
                writer.add_scalar('val/double_q_min', total_double_q_min_value / 10, epoch)
                writer.add_scalar('val/q_1_accuracy', q_1_accuracy / 10, epoch)
                writer.add_scalar('val/q_2_accuracy', q_2_accuracy / 10, epoch)
                writer.add_scalar('val/double_q_min_accuracy', double_q_min_accuracy / 10, epoch)
                writer.add_scalar('val/actor_accuracy', actor_accuracy / 10, epoch)
                num_batches = 0
                for batch in dataloader: 
                    batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch

                    # 得到 Q 值 (或者策略分布)
                    train_q_1 , train_q_2 , train_action = model(batch_pre_state)  # [batch, num_actions]
                    train_q_1_action = torch.argmax(train_q_1, dim=1)  # greedy action
                    train_q_2_action = torch.argmax(train_q_2, dim=1)
                    train_double_q_min_action = torch.argmax(torch.min(train_q_1 , train_q_2) , dim=1)
                    train_actor_action = torch.argmax(train_action, dim=1)
                    train_q_1_selected = train_q_1.gather(1, train_q_1_action.unsqueeze(1)).squeeze(1)
                    train_q_2_selected = train_q_2.gather(1, train_q_2_action.unsqueeze(1)).squeeze(1)
                    train_double_q_min_selected = torch.min(train_q_1,train_q_2).gather(1 , train_double_q_min_action.unsqueeze(1)).squeeze(1)
                    
                    train_total_q_1_value += train_q_1_selected.mean().item()
                    train_total_q_2_value += train_q_2_selected.mean().item()
                    train_total_double_q_min_value+=train_double_q_min_selected.mean().item()
                    train_q_1_accuracy += ((train_q_1_action == batch_labels).sum().item())/ batch_labels.size(0)
                    train_q_2_accuracy += ((train_q_2_action == batch_labels).sum().item())/ batch_labels.size(0)
                    train_double_q_min_accuracy+=((train_double_q_min_action == batch_labels).sum().item())/batch_labels.size(0)
                    
                    train_actor_accuracy += ((train_actor_action == batch_labels).sum().item())/ batch_labels.size(0)
                    num_batches += 1
                    if num_batches >= 10:  # 只验证10个 batch 就够了，别太频繁
                        break
                writer.add_scalar('train/total_q_1_value', train_total_q_1_value / 10, epoch)
                writer.add_scalar('train/total_q_2_value', train_total_q_2_value / 10, epoch)
                writer.add_scalar('train/train_total_double_q_min_value', train_total_double_q_min_value / 10, epoch)
                writer.add_scalar('train/q_1_accuracy', train_q_1_accuracy / 10, epoch)
                writer.add_scalar('train/q_2_accuracy', train_q_2_accuracy / 10, epoch)
                writer.add_scalar('train/actor_accuracy', train_actor_accuracy / 10, epoch)
                writer.add_scalar('train/train_double_q_min_accuracy', train_double_q_min_accuracy / 10, epoch)
            model.train()

    torch.save(model.state_dict(), f'{ckpt_dir}/shuffle_mutienv_cql_dn_combinencode_level_0_and_1_2.pth')
