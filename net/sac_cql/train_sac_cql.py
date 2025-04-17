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
from yz.net import use_combinencode_level_data_advance_stop as Data
from yz.net import use_combinencode_data as Val_Data
# from yz.net.utils import lmdb_sampler
from yz.net.utils import lmdb_sampler_advance_stop
from yz.net.sac_cql.sac_cql_fine_tune import DiscreteSAC_CQL as SAC_CQL
# from yz.val_scripts.spl import Actor

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
    
class DiscreteSAC_CQL:
    def __init__(self, model, device, learning_rate=1e-4, alpha=0.1, tau=0.005):
        self.model = model
        self.action_dim = 4
        self.target_entropy = -self.action_dim  # 目标熵（离散 SAC）
        self.target_model = AVNet(128, 4, 128, 36).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.device = device
        self.lr = learning_rate
        self.tau = tau  # 目标网络软更新系数
        self.temperature = 1.0
        self.target_action_gap = 0
        # alpha 相关参数
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)  # log_alpha 存储
        self.alpha = self.log_alpha.exp().detach() # 计算 alpha , 脱离计算图
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)  # Adam 优化器
        # cql_alpha 相关参数
        self.log_alpha_cql = torch.tensor([0.0], requires_grad=True, device="cuda")  # CQL 的 alpha 参数
        self.alpha_cql = self.log_alpha_cql.exp()  # 指数映射，确保 alpha 始终为正
        self.alpha_cql_optimizer = optim.Adam([self.log_alpha_cql], lr=self.lr)  # 使用 Adam 优化


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # def compute_loss(self, q1, q2, logits, target_q, labels):
    #     """ 计算离散 SAC + CQL 损失 """

    #     # 选取执行的动作 Q 值
    #     pdb.set_trace()
    #     a_Q1 = q1.gather(1, labels.unsqueeze(1))
    #     a_Q2 = q2.gather(1, labels.unsqueeze(1))
    #     min_q = torch.min(a_Q1, a_Q2)  # 双 Q 学习

    #     # Q-learning 目标
    #     # pdb.set_trace()
    #     q_loss = F.mse_loss(min_q, target_q.unsqueeze(1))

    #     # CQL 额外约束
    #     q_regularization = ( (torch.logsumexp(q1, dim=1).mean() - a_Q1.mean()) + \
    #                        (torch.logsumexp(q2, dim=1).mean() - a_Q2.mean()) )
        
    #     alpha_cql_loss = -torch.mean(self.log_alpha_cql.exp() * (q_regularization.detach() - self.target_action_gap))
    #     # 策略损失（离散 SAC）
    #     policy_dist = F.softmax(logits / self.temperature, dim=1)
    #     policy_loss = torch.mean(torch.sum(policy_dist * (self.alpha.detach() * torch.log(policy_dist + 1e-10) - min_q), dim=1))

    #     total_loss = q_loss +  q_regularization + policy_loss
    #     return total_loss, q_loss, q_regularization, policy_loss , alpha_cql_loss
    def compute_loss(self, q1, q2, logits, target_q, labels):
        """
        计算 Discrete SAC + CQL 的综合损失
        包括 Q loss、policy loss、CQL regularization、alpha loss 和 alpha_cql_loss
        """

        # 选取执行的动作 Q 值
        a_Q1 = q1.gather(1, labels.unsqueeze(1))
        a_Q2 = q2.gather(1, labels.unsqueeze(1))
        min_q = torch.min(a_Q1, a_Q2)  # [batch_size, 1]

        # ========================
        # 1. Q-Learning Loss
        # ========================
        q_loss = F.mse_loss(min_q, target_q.unsqueeze(1))  # Q-target loss

        # ========================
        # 2. CQL Regularization
        # ========================
        logsum_q1 = torch.logsumexp(q1, dim=1).mean()
        logsum_q2 = torch.logsumexp(q2, dim=1).mean()
        cql_regularization = 0.5 * ((logsum_q1 - q1.mean()) + (logsum_q2 - q2.mean()))

        # CQL Alpha Loss (优化 alpha_cql)
        alpha_cql_loss = -torch.mean(self.log_alpha_cql.exp() * (cql_regularization.detach() - self.target_action_gap))

        # ========================
        # 3. Policy Loss（修复版本）
        # ========================
        q_min_all = torch.min(q1, q2)  # [batch, action_dim]
        log_policy = F.log_softmax(logits / self.temperature, dim=1)  # [batch, action_dim]
        policy_dist = log_policy.exp()

        policy_loss = torch.mean(torch.sum(
            policy_dist * (self.alpha.detach() * log_policy - q_min_all),
            dim=1))

        # ========================
        # 4. Total Loss
        # ========================
        total_loss = q_loss + self.log_alpha_cql.exp() * cql_regularization + policy_loss

        return total_loss, q_loss, cql_regularization, policy_loss, alpha_cql_loss

    def train_step(self, pre_state , next_state, labels, reward, done):


        q1, q2, logits = self.model(pre_state)
        with torch.no_grad():
            # pdb.set_trace()
            next_q1, next_q2, next_logits = self.target_model(next_state)
            next_min_q = torch.min(next_q1, next_q2)
            next_policy = F.softmax(next_logits / self.temperature , dim=1)
            
            next_value = (next_policy * (next_min_q - self.alpha.detach() * torch.log(next_policy + 1e-10))).sum(dim=1)
            # pdb.set_trace()
            target_q = reward + (1 - done) * 0.99 * next_value
        total_loss, q_loss, q_regularization, policy_loss ,alpha_cql_loss= self.compute_loss(q1, q2, logits, target_q , labels)
        
        # _,_, alp_logits = self.model(pre_audio, pre_visual)
        # 计算 entropy 并优化 alpha
        policy_dist = F.softmax(logits / self.temperature, dim=1)
        entropy = -torch.sum(policy_dist * torch.log(policy_dist + 1e-10), dim=1).mean()
        # alpha_loss = -torch.mean(self.log_alpha.exp() * (entropy.detach() + self.target_entropy))
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy.detach() - self.target_entropy))
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha_cql_optimizer.zero_grad()
        alpha_cql_loss.backward()
        self.alpha_cql_optimizer.step()
        # 更新 alpha
        self.alpha = self.log_alpha.exp()
        self.alpha_cql = self.log_alpha_cql.exp()
        # 目标网络软更新
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return total_loss.item(), q_loss.item(), q_regularization.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item() ,alpha_cql_loss.item() , self.alpha_cql.item()


def train(ckpt_dir):
    # database_dir = agent_config.DATABASE_DIR
    database_dir = agent_config.RELATIVE_DATABASE_DIR
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVNet(128, 4, 128, 36).to(device)
    model.train()
    cql = SAC_CQL(model, device)
    episode = 0
    database_path_stop = [os.path.join(f'{database_dir}/mutienv_data_combinencode_advance_stop_level/' , i) \
        for i in os.listdir(f'{database_dir}/mutienv_data_combinencode_advance_stop_level/')]
    
    database_path = [os.path.join(f'{database_dir}/new_key_shffule_mutienv_data_combinencode_level/' , i) \
        for i in os.listdir(f'{database_dir}/new_key_shffule_mutienv_data_combinencode_level/')]
    init_sample_dict = {
        'level_0':1,
        'level_1':0,
        'level_2':0,
    }
    sampler = lmdb_sampler_advance_stop(database_path,stop_database_path=database_path_stop, shuffle=True)
    sampler.sample_data(sample=init_sample_dict)
    dataset = Data(database_path , database_path_stop)
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
        # pdb.set_trace()
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
            
            if episode % 1000 == 0 :
                logging.info(f"Epoch {epoch} , episode : {episode} ,time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
            if episode % 10000 == 0 and episode != 0:
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
                
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")

    torch.save(model.state_dict(), f'{ckpt_dir}/shuffle_mutienv_cql_dn_combinencode_level_0_and_1_2.pth')

def module_test_spl_val(chpt_dir):
    # database_dir = agent_config.DATABASE_DIR
    database_dir = agent_config.RELATIVE_DATABASE_DIR
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AVNet(128, 4, 128, 36).to(device)
    model.train()
    cql = SAC_CQL(model, device)
    episode = 0
    database_path_stop = [os.path.join(f'{database_dir}/mutienv_data_combinencode_advance_stop_level/' , i) \
        for i in os.listdir(f'{database_dir}/mutienv_data_combinencode_advance_stop_level/')]
    
    database_path = [os.path.join(f'{database_dir}/new_key_shffule_mutienv_data_combinencode_level/' , i) \
        for i in os.listdir(f'{database_dir}/new_key_shffule_mutienv_data_combinencode_level/')]
    init_sample_dict = {
        'level_0':1,
        'level_1':0,
        'level_2':0,
    }
    sampler = lmdb_sampler_advance_stop(database_path,stop_database_path=database_path_stop, shuffle=True)
    sampler.sample_data(sample=init_sample_dict)
    dataset = Data(database_path , database_path_stop)
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
            
        if epoch % 1 == 0:
           num_actors = 10
           num_gpus = 1
           ray.init(num_cpus=10, num_gpus=num_gpus)
           actors = [Actor.remote(cql) for i in range(num_actors)]
           
           import pdb; pdb.set_trace()
           token_id = [actor.val.remote() for actor in actors]
           result_measurement = ray.get(token_id)
           print(result_measurement)
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    # base_dir = agent_config.EXPERIMENTS_DIR
    base_dir = agent_config.RELATIVE_EXPERIMENTS_DIR
    time_stamp = "{0:%Y-%m-%d~%H-%M-%S}".format(datetime.now())
    loss_dir = base_dir +'/loss/'  + time_stamp
    log_dir = base_dir + '/log/'
    ckpt_dir = base_dir + '/ckpt/' + time_stamp
    train_message = base_dir + 'train.log'
    with open(train_message , 'a') as f:
        f.write(f'\n{time_stamp} , message: 用最开始的那个model了，发现这个改来改去还是最开始的acc高')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    writer = SummaryWriter(loss_dir)
    logging.basicConfig(filename=f'{log_dir}/{time_stamp}.log', level=logging.INFO,filemode='a')
    train(ckpt_dir)
    # module_test_spl_val(ckpt_dir)