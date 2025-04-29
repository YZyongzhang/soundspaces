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

# from yz.net import use_combinencode_level_data as Data
# from yz.net import use_combinencode_level_data_advance_stop as Data
from yz.net import use_combinencode_data as Val_Data
# # from yz.net.utils import lmdb_sampler
from yz.net import use_combinencode_level_data_time_seq as Data
from yz.net.utils import lmdb_sampler_time_seq as lmdb_sampler
from yz.net.sac_cql.SAC_CQL import CQLSAC as SAC_CQL
from yz.net.sac_cql.SAC_CQL import Critic_Actor_GRU as Critic_Actor



def Train(model,dataloader,episode,writer):
        for batch_data in dataloader:
            batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch_data
            # pdb.set_trace()
            batch_done = batch_done.to(device)
                
            loss_dict = model.learn(
                (batch_pre_state ,batch_next_state, batch_labels, batch_reward, batch_done),epoch
            )
            print(f'epoch:{epoch} , episode {episode}' + ",".join([f"{k}: {v}" for k, v in loss_dict.items()]))
            for name, item in loss_dict.items():
                writer.add_scalar(f'loss/{name}', item, episode)
            episode += 1
            return loss_dict
            
        if epoch % 100 == 0 and epoch != 0:
            torch.save(cql.state_dict(), f'{ckpt_dir}/shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_{episode}_{epoch}.pth')
            
        # if epoch % 1 == 0:
        #     cql.eval()
        #     train_total_q_1_value = 0
        #     train_total_q_2_value = 0
        #     train_total_double_q_min_value=0
        #     train_q_1_accuracy = 0
        #     train_q_2_accuracy = 0
        #     train_double_q_min_accuracy=0
        #     train_actor_accuracy = 0
        #     num_batches = 0
        #     with torch.no_grad():
        #         for batch in dataloader: 
        #             batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch

        #             batchsize, time_seq ,_ = batch_pre_state.shape
        #             batch_pre_state,_ = cql.gru(batch_pre_state)
        #             batch_pre_state=batch_pre_state.reshape(batchsize*time_seq,-1)
        #             train_q_1 =cql.critic1(batch_pre_state)
        #             train_q_2 =cql.critic2(batch_pre_state)
        #             train_action = cql.actor_local(batch_pre_state)  # [batch, num_actions]
                    
        #             train_q_1_action = torch.argmax(train_q_1, dim=1)  # greedy action
        #             train_q_2_action = torch.argmax(train_q_2, dim=1)
        #             train_double_q_min_action = torch.argmax(torch.min(train_q_1 , train_q_2) , dim=1)
        #             train_actor_action = torch.argmax(train_action, dim=1)
        #             train_q_1_selected = train_q_1.gather(1, train_q_1_action.unsqueeze(1)).squeeze(1)
        #             train_q_2_selected = train_q_2.gather(1, train_q_2_action.unsqueeze(1)).squeeze(1)
        #             train_double_q_min_selected = torch.min(train_q_1,train_q_2).gather(1 , train_double_q_min_action.unsqueeze(1)).squeeze(1)
        #             # pdb.set_trace()
        #             train_total_q_1_value += train_q_1_selected.mean().item()
        #             train_total_q_2_value += train_q_2_selected.mean().item()
        #             train_total_double_q_min_value+=train_double_q_min_selected.mean().item()
        #             # pdb.set_trace()
        #             train_q_1_accuracy += ((train_q_1_action == batch_labels).sum().item())/ batch_labels.size(0)
        #             train_q_2_accuracy += ((train_q_2_action == batch_labels).sum().item())/ batch_labels.size(0)
        #             train_double_q_min_accuracy+=((train_double_q_min_action == batch_labels).sum().item())/batch_labels.size(0)
                    
        #             train_actor_accuracy += ((train_actor_action == batch_labels).sum().item())/ batch_labels.size(0)
        #             num_batches += 1
        #             if num_batches >= 10:  # 只验证10个 batch 就够了，别太频繁
        #                 break
        #         train_actor_action_counts = torch.bincount(train_actor_action, minlength=4).float()
        #         train_actor_action_probs = train_actor_action_counts / train_actor_action_counts.sum()
        #         for i in range(4):
        #             writer.add_scalar(f"Actor/Action_{i}_prob", train_actor_action_probs[i].item(), epoch)
                
        #         writer.add_scalar('train/total_q_1_value', train_total_q_1_value / 10, epoch)
        #         writer.add_scalar('train/total_q_2_value', train_total_q_2_value / 10, epoch)
        #         writer.add_scalar('train/train_total_double_q_min_value', train_total_double_q_min_value / 10, epoch)
        #         writer.add_scalar('train/q_1_accuracy', train_q_1_accuracy / 10, epoch)
        #         writer.add_scalar('train/q_2_accuracy', train_q_2_accuracy / 10, epoch)
        #         writer.add_scalar('train/actor_accuracy', train_actor_accuracy / 10, epoch)
        #         writer.add_scalar('train/train_double_q_min_accuracy', train_double_q_min_accuracy / 10, epoch)
        #     cql.train()

    torch.save(cql.state_dict(), f'{ckpt_dir}/shuffle_mutienv_cql_dn_combinencode_level_0_and_1_2.pth')
