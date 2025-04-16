import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import os , math
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
import time
import pdb
import sys ,copy
sys.path.append('/home/getuanhui/project/sound-spaces')
from yz.config import agent_config

# from yz.net import use_combinencode_level_data as Data
from yz.net import use_combinencode_level_data_advance_stop as Data
from yz.net import use_combinencode_data as Val_Data
# from yz.net.utils import lmdb_sampler
from yz.net.utils import lmdb_sampler_advance_stop


class Q_Net(nn.Module):
    def __init__(self, out_put):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_put = out_put


        self.Q_net1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )

    def forward(self, combinencode):
        q1 = self.Q_net1(combinencode)
        return q1
class Policy_Net(nn.Module):
    def __init__(self, out_put):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_put = out_put

        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_put)
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, combinencode):
        x = self.policy_net(combinencode)
        action_probs = self.softmax(x)
        return action_probs
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC, self).__init__()
        self.action_size = 4

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -self.action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # Actor Network 

        self.actor_local = Policy_Net(self.action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Q_Net(self.action_size).to(device)
        self.critic2 = Q_Net(self.action_size).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Q_Net(self.action_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Q_Net(self.action_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        self.softmax = nn.Softmax(dim=-1)

    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, pre_state , next_state, labels, reward, done):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = pre_state , labels ,reward, next_state, done

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)
            # pdb.set_trace()
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1)) 


        # Compute critic loss
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        
        # pdb.set_trace()
        q1_ = q1.gather(1, actions.long().unsqueeze(1))
        q2_ = q2.gather(1, actions.long().unsqueeze(1))
        
        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets.unsqueeze(1))
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets.unsqueeze(1))
        
        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        # return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()
        return {
            'actor_loss':actor_loss.item(),
            'alpha_loss':alpha_loss.item(),
            'critic1_loss':critic1_loss.item(),
            'critic2_loss':critic2_loss.item(),
            'cql1_scaled_loss':cql1_scaled_loss.item(),
            'cql2_scaled_loss':cql2_scaled_loss.item(),
            "current_alpha":current_alpha, 
            'cql_alpha_loss':cql_alpha_loss.item(), 
            'cql_alpha_loss':cql_alpha.item()
        }
    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


def train(ckpt_dir):
    # database_dir = agent_config.DATABASE_DIR
    database_dir = agent_config.RELATIVE_DATABASE_DIR
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_star = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = AVNet(128, 4, 128, 36).to(device)
    # model.train()
    # cql = DiscreteSAC_CQL(model, device)
    cql = CQLSAC(device)
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
                
            # total_loss, q_loss, q_regularization, policy_loss ,alpha_loss, alpha= cql.learn(
            #     batch_pre_state ,batch_next_state, batch_labels, batch_reward, batch_done
            # )
            # writer.add_scalar('Loss/total_loss', total_loss, episode)
            # writer.add_scalar('Loss/q_loss', q_loss, episode)
            # writer.add_scalar('Loss/q_regularization', q_regularization, episode)
            # writer.add_scalar('Loss/policy_loss', policy_loss, episode)
            # writer.add_scalar('Loss/alpha_loss', alpha_loss, episode)
            # writer.add_scalar('Loss/alpha', alpha, episode)
            # # writer.add_scalar('Loss/cql_alpha_loss', cql_alpha_loss, episode)
            # # writer.add_scalar('Loss/cql_alpha', cql_alpha, episode)
            # print(f'Epoch {epoch} , episode : {episode}: Loss {total_loss}, Q-Loss {q_loss}, CQL-Reg {q_regularization}, Policy Loss {policy_loss}')
            loss_dict = cql.learn(
                batch_pre_state ,batch_next_state, batch_labels, batch_reward, batch_done
            )
            print(f'epoch:{epoch} , episode {episode}' + ",".join([f"{k}: {v}" for k, v in loss_dict.items()]))
            for name, item in loss_dict.items():
                writer.add_scalar('loss/name', item, episode)
            episode += 1
            
            if episode % 1000 == 0 :
                logging.info(f"Epoch {epoch} , episode : {episode} ,time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")
            if episode % 10000 == 0 and episode != 0:
                torch.save(cql.state_dict(), f'{ckpt_dir}/shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_{episode}_{epoch}.pth')
            
        if epoch % 2 == 0:
            cql.critic1.eval()
            cql.critic2.eval()
            cql.actor_local.eval()
            total_q_1_value = 0
            total_q_2_value = 0
            train_total_q_1_value = 0
            train_total_q_2_value = 0
            q_1_accuracy = 0
            q_2_accuracy = 0
            actor_accuracy = 0
            train_q_1_accuracy = 0
            train_q_2_accuracy = 0
            train_actor_accuracy = 0
            num_batches = 0
            with torch.no_grad():
                for val_batch in val_dataloader: 
                    batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = val_batch

                    # 得到 Q 值 (或者策略分布)
                    q_1 = cql.critic1(batch_pre_state)  # [batch, num_actions]
                    q_2 = cql.critic2(batch_pre_state)
                    action = cql.actor_local(batch_pre_state)
                    q_1_action = torch.argmax(q_1, dim=1)  # greedy action
                    q_2_action = torch.argmax(q_2, dim=1)
                    actor_action = torch.argmax(action, dim=1)
                    q_1_selected = q_1.gather(1, q_1_action.unsqueeze(1)).squeeze(1)
                    q_2_selected = q_2.gather(1, q_2_action.unsqueeze(1)).squeeze(1)
                    total_q_1_value += q_1_selected.mean().item()
                    total_q_2_value += q_2_selected.mean().item()
                    q_1_accuracy += ((q_1_action == batch_labels).sum().item())/ batch_labels.size(0)
                    q_2_accuracy += ((q_2_action == batch_labels).sum().item())/ batch_labels.size(0)
                    actor_accuracy += ((actor_action == batch_labels).sum().item())/ batch_labels.size(0)
                    num_batches += 1
                    if num_batches >= 10:  # 只验证10个 batch 就够了，别太频繁
                        break
                writer.add_scalar('val/total_q_1_value', total_q_1_value / 10, epoch)
                writer.add_scalar('val/total_q_2_value', total_q_2_value / 10, epoch)
                writer.add_scalar('val/q_1_accuracy', q_1_accuracy / 10, epoch)
                writer.add_scalar('val/q_2_accuracy', q_2_accuracy / 10, epoch)
                writer.add_scalar('val/actor_accuracy', actor_accuracy / 10, epoch)
                num_batches = 0
                for batch in dataloader: 
                    batch_pre_state , batch_next_state, batch_done, batch_reward, batch_labels = batch

                    # 得到 Q 值 (或者策略分布)
                    train_q_1 = cql.critic1(batch_pre_state)  # [batch, num_actions]
                    train_q_2 = cql.critic2(batch_pre_state)
                    train_action = cql.actor_local(batch_pre_state)
                    train_q_1_action = torch.argmax(train_q_1, dim=1)  # greedy action
                    train_q_2_action = torch.argmax(train_q_2, dim=1)
                    train_actor_action = torch.argmax(train_action, dim=1)
                    train_q_1_selected = train_q_1.gather(1, train_q_1_action.unsqueeze(1)).squeeze(1)
                    train_q_2_selected = train_q_2.gather(1, train_q_2_action.unsqueeze(1)).squeeze(1)
                    train_total_q_1_value += train_q_1_selected.mean().item()
                    train_total_q_2_value += train_q_2_selected.mean().item()
                    train_q_1_accuracy += ((train_q_1_action == batch_labels).sum().item())/ batch_labels.size(0)
                    train_q_2_accuracy += ((train_q_2_action == batch_labels).sum().item())/ batch_labels.size(0)
                    train_actor_accuracy += ((train_actor_action == batch_labels).sum().item())/ batch_labels.size(0)
                    num_batches += 1
                    if num_batches >= 10:  # 只验证10个 batch 就够了，别太频繁
                        break
                writer.add_scalar('train/total_q_1_value', train_total_q_1_value / 10, epoch)
                writer.add_scalar('train/total_q_2_value', train_total_q_2_value / 10, epoch)
                writer.add_scalar('train/q_1_accuracy', train_q_1_accuracy / 10, epoch)
                writer.add_scalar('train/q_2_accuracy', train_q_2_accuracy / 10, epoch)
                writer.add_scalar('train/actor_accuracy', train_actor_accuracy / 10, epoch)
            cql.critic1.train()
            cql.critic2.train()
            cql.actor_local.train()
                
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"begin time : {current_time_str} now time :{now_time}")
        logging.info(f"time : {(time.time()  - time_star) // 60 } m {(time.time() - time_star) % 60 } s")

    torch.save(cql.state_dict(), f'{ckpt_dir}/shuffle_mutienv_cql_dn_combinencode_level_0_and_1_2.pth')

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
        f.write(f'\n{time_stamp} , message: 将alpha loss 中的一个符号改一下，我觉得这个应该有问题。')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    writer = SummaryWriter(loss_dir)
    logging.basicConfig(filename=f'{log_dir}/{time_stamp}.log', level=logging.INFO,filemode='a')
    train(ckpt_dir)