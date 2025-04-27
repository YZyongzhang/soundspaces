import lmdb , pickle , pdb
import torch
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torch.utils.data import Sampler
import os , re
import random
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
import time
import pdb
from yz.net import AVFNet
from yz.config import agent_config

class USE_COMBINENCODE_DATA(Dataset):
    def __init__(self,database_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
        self.databases = database_path
        self.envs = lmdb.open(self.databases , readonly = True)
        self.states = self.envs.stat()
        self.txns = self.envs.begin()
    def __len__(self):
        # pdb.set_trace()
        num = 0
        num+= self.states['entries']
        return num
    def __getitem__(self,idx):
        # print(self.files[idx])
        # print(idx)
        key = f'key_{idx}'.encode()
        # print(key)
        value = self.txns.get(key)
        batch_data = pickle.loads(value)
        return self.deal_data(batch_data)
    
    def deal_data(self,batch_data):
        pre_state , next_state , done , reward , action = batch_data.values()
        # pdb.set_trace()
        return pre_state , next_state , done , reward,action

class TO_STATEENCODE_DATABASE:
    def __init__(self):
        self.model_path = f'/home/getuanhui/project/sound-spaces/yz/data/checkpoint/acmcheckpoint/avf_muti_env_90000.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.combinencode_base_path = agent_config.RELATIVE_DATABASE_DIR

    # def gen_database(self , database_path ,want_gen_database,level):
    #     # database_path: we generate episode database
    #     # want_gen_database: we want put the encode_database path
    #     encode_env = lmdb.open(f"{self.combinencode_base_path}/{want_gen_database}/combinencode_level_{level}" , map_size= 1024 * 1024 * 1024 * 50)
    #     encode_txn = encode_env.begin(write=True)
    #     mydata  = USE_LMBD_DATABASE(database_path)
    #     keyid = 0
    #     for idx in tqdm(range(mydata.__len__()) , desc= 'load data'):
    #         pre_audio,pre_visual, next_audio, next_visual,done,reward,action = mydata.__getitem__(idx)
    #         pre_state_combinencode = self.model(pre_audio , pre_visual)
    #         next_state_combinencode = self.model(next_audio , next_visual)
    #         if pre_state_combinencode.shape[0] == next_state_combinencode.shape[0]:
    #             for batch_idx in range(pre_state_combinencode.shape[0]):
    #                 key = f'{level}_{keyid}'.encode()
    #                 data = {
    #                     'pre_state':pre_state_combinencode[batch_idx],
    #                     'next_state':next_state_combinencode[batch_idx],
    #                     'done': torch.tensor(done[batch_idx]),
    #                     'reward' :reward[batch_idx], 
    #                     'action' :action[batch_idx],
    #                 }
    #                 value = pickle.dumps(data)
    #                 encode_txn.put(key , value)
    #                 if keyid % 500 == 0:
    #                     encode_txn.commit()
    #                     encode_txn = encode_env.begin(write=True)
    #                 keyid+=1
    #         else:
    #             print('shape error!')
    #     print(f"finally key id {keyid}")
    #     encode_txn.commit()
    #     encode_env.close()
    def gen_database(self , database_path ,want_gen_database,level):
        # database_path: we generate episode database
        # want_gen_database: we want put the encode_database path
        # gen episode database
        encode_env = lmdb.open(f"{self.combinencode_base_path}/{want_gen_database}/combinencode_level_{level}" , map_size= 1024 * 1024 * 1024 * 50)
        encode_txn = encode_env.begin(write=True)
        mydata  = USE_LMBD_DATABASE(database_path)
        keyid = 0
        for idx in tqdm(range(mydata.__len__()) , desc= 'load data'):
            pre_audio,pre_visual, next_audio, next_visual,done,reward,action = mydata.__getitem__(level , idx)
            pre_state_combinencode = self.model(pre_audio , pre_visual)
            next_state_combinencode = self.model(next_audio , next_visual)
            if pre_state_combinencode.shape[0] == next_state_combinencode.shape[0]:
                key = f'{level}_{keyid}'.encode()
                data = {
                    'pre_state':pre_state_combinencode,
                    'next_state':next_state_combinencode,
                    'done': torch.tensor(done,device=self.device),
                    'reward' :reward.to(self.device),
                    'action' :action.to(self.device)
                }
                value = pickle.dumps(data)
                encode_txn.put(key , value)
                if keyid % 500 == 0:
                    encode_txn.commit()
                    encode_txn = encode_env.begin(write=True)
                keyid+=1
            else:
                print('shape error!')
        print(f"finally key id {keyid}")
        encode_txn.commit()
        encode_env.close()
            
class TRANS_TO_DATABASE_FROM_RAWDATA:
    def __init__(self , store_path):
        self.data_base_path = agent_config.RELATIVE_DATABASE_DIR
        self.database_store_path = store_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def tran_to_lmdb(self , levels_data):
        for level ,items in levels_data.items():
            # key : level(0,1,2) , items (files)
            env = lmdb.open(f"{self.data_base_path}/{self.database_store_path}/muti_env_data_{level}" , map_size= 1024 * 1024 * 1024 * 150)
            txn = env.begin(write=True)
        
            for episode , file in enumerate(tqdm(items , desc='load data')):
                # file_path  obsolutly path
                with open(file , 'rb') as f:
                    data = pickle.load(f)
                dealed_data = self.deal_data(data)
                key = f'{level}_{episode}'.encode()
                value = pickle.dumps(dealed_data)
                txn.put(key , value)
                
                if episode % 100 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            txn.commit()
            env.close()
    def fusion_data(self , raw_datas):
        
        levels_data = {
            'level0':[],
            'level1':[],
            'level2':[]
        }
        # evey path is obsolutly path like /home/getunahui/-----
        # raw_data = [raw1 ,raw2 , raw3]
        # [raw: level0 ,level1 ,level2]
        
        for raw_data in raw_datas:
            # raw_data = path
            levels = sorted(os.listdir(raw_data), key=lambda x: int(''.join(filter(str.isdigit, x))))
            # listdir raw_data = [level0 , level1 ,level2]
            # get obsolutlt path
            levels_path = [os.path.join(raw_data , i) for i in levels]
            # get evey levels files for levels_data
            for level_name , level_path in zip(levels , levels_path):
                temp = [os.path.join(level_path , i)  for i in os.listdir(level_path)]
                levels_data[level_name].extend(temp)
        return levels_data
    def get_trans(self , raw_data):
        # raw_data = [raw1 ,raw2 , raw3]
        # [raw: level0 ,level1 ,level2]
        levels_data = self.fusion_data(raw_data)
        """
        levels_data = {
            'level0':obsolutly filepath,
            'level1':obsolutly filepath,
            'level2':obsolutly filepath
        }
        """
        self.tran_to_lmdb(levels_data)
    def deal_data(self,batch_data):
        self.error = []
        seq = batch_data[0]
        current_done = batch_data[2]
        frist_state = batch_data[3]
        audio_ = self.get_data(seq,'audio')
        action_ = self.get_data(seq,'rl_pred')
        visual_ = self.get_data(seq,'camera')
        reward_ = self.get_data(seq,'reward')
        step = self.get_data(seq , 'step')
        done  , next_audio , next_visual , action , reward = [] , [] , [] , [] , []

        index  = max(step)
        reward = reward_[:index]
        action = action_[:index]

        next_audio = audio_[1:index+1] 
        next_visual = visual_[1:index+1]
        pre_audio  = audio_[:index]
        pre_visual = visual_[:index]
        done = [1 if d[0] else 0 for d in current_done]
        
        # 收集碰撞数据得时候发现最后一个step多了出来，所以这里我就之间去除最后一个数据
        # reward = reward_[:index-1]
        # action = action_[:index-1]

        # next_audio = audio_[1:index] 
        # next_visual = visual_[1:index]
        # pre_audio  = audio_[:index-1]
        # pre_visual = visual_[:index-1]
        # done = [1 if d[0] else 0 for d in current_done[:-1]]
        
        return pre_audio,pre_visual, next_audio, next_visual,done,reward,action
    def get_data(self , data , name):
        d = list()
        for i in range(len(data)):
            d.extend(self.totensor(data[i][name]))
        return torch.stack(d)
    def totensor(self,data):
        temp = torch.from_numpy(data).to(self.device)
        return temp[1:]
        
class USE_COMBINENCODE_LEVEL_DATA(Dataset):
    def __init__(self, database_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
        self.databases = database_path
        self.envs = [(lmdb.open(database , readonly = True) , database[-7:]) for database in self.databases]
        self.error = list()
        
    def __getitem__(self,idx_tuple):
        # print(idx_tuple)
        env_name , key = idx_tuple
        txn = self.map(env_name)
        value = txn.get(key)
        if value is None:
            print(f"error {key}")
            self.error.append(key)
            return None 
        batch_data = pickle.loads(value)
        return self.deal_data(batch_data)
    
    def deal_data(self,batch_data):
        pre_state , next_state , done , reward , action = batch_data.values()
        # pdb.set_trace()
        return pre_state , next_state , done , reward,action
    def map(self,env_name):
        # print(env_name)
        if env_name == self.envs[0][1]:
            return self.envs[0][0].begin()
        elif env_name == self.envs[1][1]:
            return self.envs[1][0].begin()
        elif env_name == self.envs[2][1]:
            return self.envs[2][0].begin()
    
class USE_COMBINENCODE_LEVEL_DATA_ADVANCE_STOP(Dataset):
    def __init__(self, database_path , stop_database_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
        self.databases = database_path
        self.stop_database_path = stop_database_path
        self.envs = [(lmdb.open(database , readonly = True) , database[-7:]) for database in self.databases]
        self.envs_stop = [(lmdb.open(stop_database , readonly = True) , f'{stop_database[-7:]}_stop') for stop_database in self.stop_database_path]
        self.error = list()
        
    def __getitem__(self,idx_tuple):
        # print(idx_tuple)
        env_name , key = idx_tuple
        txn = self.map(env_name)
        value = txn.get(key)
        if value is None:
            print(f"error {key}")
            self.error.append(key)
            return None 
        batch_data = pickle.loads(value)
        return self.deal_data(batch_data)
    
    def deal_data(self,batch_data):
        pre_state , next_state , done , reward , action = batch_data.values()
        # pdb.set_trace()
        return pre_state , next_state , done , reward,action
    def map(self,env_name):
        # print(env_name)
        if env_name == self.envs[0][1]:
            return self.envs[0][0].begin()
        elif env_name == self.envs[1][1]:
            return self.envs[1][0].begin()
        elif env_name == self.envs[2][1]:
            return self.envs[2][0].begin()
        elif env_name == self.envs_stop[0][1]:
            return self.envs_stop[0][0].begin()
        elif env_name == self.envs_stop[1][1]:
            return self.envs_stop[1][0].begin()
        elif env_name == self.envs_stop[2][1]:
            return self.envs_stop[2][0].begin()

class LMDB_SAMPLER(Sampler):
    def __init__(self, database_path, shuffle=False):
        self.databases = database_path
        self.envs = [lmdb.open(database , readonly = True) for database in self.databases]
        self.states = [env.stat() for env in self.envs]
        self.txns = [env.begin() for env in self.envs]
        self.pattern = re.compile(r'level_(\d+)')
        self.idx_data = list()
        self.shuffle = shuffle
        self.map_virtual_idx()
        self.iter_idx = list()
    
    def map_virtual_idx(self):
        self.idx_data = {
            'level_0': [],
            'level_1': [],
            'level_2': []
        }

        for txn in self.txns:
            cursor = txn.cursor()
            for key, _ in tqdm(cursor ,desc='load all key'):
                key_str = key.decode()
                match = self.pattern.search(key_str)
                if match:
                    level = match.group()  # 比如 'level_0'
                    if level in self.idx_data:
                        self.idx_data[level].append((level, key))
                    else:
                        raise ValueError(f"Unrecognized level: {level}")
                else:
                    raise ValueError(f"Key pattern not matched: {key_str}")

        if self.shuffle:
            for level in self.idx_data:
                random.shuffle(self.idx_data[level])
            
    def sample_data(self, sample):
        sampled_data = []

        for level, num_rate in sample.items():
            num_samples = int(len(self.idx_data[level]) * num_rate)
            if level in self.idx_data:
                if num_samples > len(self.idx_data[level]):
                    raise ValueError(f"Not enough samples in {level}: requested {num_samples}, available {len(self.idx_data[level])}")
                sampled_data.extend(self.idx_data[level][:num_samples])
            else:
                raise KeyError(f"Invalid level '{level}' not found in idx_data. Available levels: {list(self.idx_data.keys())}")
        
        if self.shuffle:
            random.shuffle(sampled_data)
        self.iter_idx = sampled_data

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.iter_idx)
        return iter(self.iter_idx)
    def __len__(self):
        return len(self.iter_idx)

class LMDB_SAMPLER_ADVANCE_STOP(Sampler):
    def __init__(self, database_path, stop_database_path , shuffle=False):
        self.databases = database_path
        self.stop_database_path = stop_database_path
        
        self.envs = [lmdb.open(database , readonly = True) for database in self.databases]
        self.states = [env.stat() for env in self.envs]
        self.txns = [env.begin() for env in self.envs]
        
        self.stop_envs = [lmdb.open(stop_database , readonly = True) for stop_database in self.stop_database_path]
        self.stop_states = [env.stat() for env in self.stop_envs]
        self.stop_txns = [env.begin() for env in self.stop_envs]
        
        
        self.pattern = re.compile(r'level_(\d+)')
        self.idx_data = list()
        self.shuffle = shuffle
        self.map_virtual_idx()
        self.iter_idx = list()
    
    def map_virtual_idx(self):
        self.idx_data = {
            'level_0': [],
            'level_1': [],
            'level_2': []
        }

        for txn in self.txns:
            cursor = txn.cursor()
            for key, _ in tqdm(cursor ,desc='load all key'):
                key_str = key.decode()
                match = self.pattern.search(key_str)
                if match:
                    level = match.group()  # 比如 'level_0'
                    if level in self.idx_data:
                        self.idx_data[level].append((level, key))
                    else:
                        raise ValueError(f"Unrecognized level: {level}")
                else:
                    raise ValueError(f"Key pattern not matched: {key_str}")\
                        
        for stop_txn in self.stop_txns:
            cursor = stop_txn.cursor()
            for key, _ in tqdm(cursor ,desc='load all key'):
                key_str = key.decode()
                match = self.pattern.search(key_str)
                if match:
                    level = match.group()  # 比如 'level_0'
                    if level in self.idx_data:
                        self.idx_data[level].append((f'{level}_stop', key))
                    else:
                        raise ValueError(f"Unrecognized level: stop_{level}")
                else:
                    raise ValueError(f"Key pattern not matched: stop_{key_str}")

        if self.shuffle:
            for level in self.idx_data:
                random.shuffle(self.idx_data[level])
            
    def sample_data(self, sample):
        sampled_data = []

        for level, num_rate in sample.items():
            num_samples = int(len(self.idx_data[level]) * num_rate)
            if level in self.idx_data:
                if num_samples > len(self.idx_data[level]):
                    raise ValueError(f"Not enough samples in {level}: requested {num_samples}, available {len(self.idx_data[level])}")
                sampled_data.extend(self.idx_data[level][:num_samples])
            else:
                raise KeyError(f"Invalid level '{level}' not found in idx_data. Available levels: {list(self.idx_data.keys())}")
        
        if self.shuffle:
            random.shuffle(sampled_data)
        self.iter_idx = sampled_data

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.iter_idx)
        return iter(self.iter_idx)
    def __len__(self):
        return len(self.iter_idx)

class USE_LMBD_DATABASE(Dataset):
    def __init__(self,database_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
        self.database = database_path
        self.env = lmdb.open(self.database , readonly = True)
        self.state = self.env.stat()
        self.txn = self.env.begin()
    def __len__(self):
        return self.state['entries']
    def __getitem__(self,level,idx):
        # pdb.set_trace()
        # print(self.files[idx])
        key = f'{level}_{idx}'.encode()
        value = self.txn.get(key)
        batch_data = pickle.loads(value)
        return batch_data

class USE_COMBINENCODE_LEVEL_DATA_TIME_SEQ(Dataset):
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
        self.error = list()
        
    def __getitem__(self,idx_tuple):
        # idx_tuple (path , idx)
        env_path , key = idx_tuple
        # print(env_path)
        txn = self.map(env_path)
        value = txn.get(key)
        if value is None:
            print(f"error {key}")
            self.error.append(key)
            return None 
        batch_data = pickle.loads(value)
        return self.deal_data(batch_data)
    
    def deal_data(self,batch_data):
        pre_state , next_state , done , reward , action = batch_data.values()
        # pdb.set_trace()
        return pre_state , next_state , done , reward,action
    def map(self,env_name):
        env = lmdb.open(env_name , readonly = True)
        return env.begin()

class LMDB_SAMPLER_TIME_SEQ(Sampler):
    """
    dababase [------/level0,level1,level2 | ------------]
    """
    def __init__(self, database_ , shuffle=False):
        self.databases = database_
        
        self.pattern = re.compile(r'level(\d+)')
        
        s_levels = self._s_levels(self.databases)
        
        s_keys = self._s_keys(s_levels)
        
        self.idx_data = s_keys
        
        self.shuffle = shuffle
        
        self.shuffle_s_keys()
        
        self.iter_idx = list()
    def _s_levels(self,databases):
        # match level
        # sum files path
        """
        databases like :
        ['/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/muti_env_advance_stop_encode',
        '/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/muti_env_crushed_encode'
        , '/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/muti_env_encode']
        """
        # pdb.set_trace()
        s_levels = list()
        for database in databases:
            # database [--------/~level0,level1,level2]
            levels_p_l = [os.path.join(database ,i) for i in os.listdir(database)]
            # levels_p_l [---------/level0 , ----------/level1, --------level2]
            s_levels.append(sorted(levels_p_l , key=lambda x: int(''.join(filter(str.isdigit, x)))))
            # sorted 
        return s_levels
    
    def _s_keys(self, s_levels):
        # s_levels 
        """
        [
            '-----/level0','----level1',''
            '','',''
        ]
        """
        s_keys = {
            'level0':[],
            'level1':[],
            'level2':[]
        }
        for level_tuple in zip(*s_levels):
            # level_tuple ('level0' ,'level0' , 'level0' ---------)
            for elm in level_tuple:
                # elm = '/------------/level0'
                # pdb.set_trace()
                env = lmdb.open(elm , readonly = True)
                txn = env.begin()
                cursor = txn.cursor()
                # traverse the key in txn
                for key , _ in tqdm(cursor , desc='load all key'):
                    key_str = key.decode()
                    match = self.pattern.search(key_str)
                    if match:
                        level = match.group()  # 比如 'level0_0''level0_1'
                        if level in s_keys:
                            # [(path , key)(path,key)-------]
                            s_keys[level].append((elm, key))
                        else:
                            raise ValueError(f"Unrecognized level: {level}")
                    else:
                        raise ValueError(f"Key pattern not matched: {key_str}")
                    
        return s_keys
      
    
    def shuffle_s_keys(self):

        if self.shuffle:
            for level in self.idx_data:
                random.shuffle(self.idx_data[level])
            
    def sample_data(self, sample):
        """
        sample_dict =   {
        'level_0':1, rate 0~1
        'level_1':0,
        'level_2':0,
        }
        """
        sampled_data = []

        for level, num_rate in sample.items():
            num_samples = int(len(self.idx_data[level]) * num_rate)
            if level in self.idx_data:
                if num_samples > len(self.idx_data[level]):
                    raise ValueError(f"Not enough samples in {level}: requested {num_samples}, available {len(self.idx_data[level])}")
                sampled_data.extend(self.idx_data[level][:num_samples])
            else:
                raise KeyError(f"Invalid level '{level}' not found in idx_data. Available levels: {list(self.idx_data.keys())}")
        
        if self.shuffle:
            random.shuffle(sampled_data)
        self.iter_idx = sampled_data

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.iter_idx)
        return iter(self.iter_idx)
    def __len__(self):
        return len(self.iter_idx)


class _split_T():
    def __init__(self , store_path):
        self.data_base_path = agent_config.RELATIVE_DATABASE_DIR
        self.database_store_path = store_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def tran_to_lmdb(self , levels_data):
        for level ,items in levels_data.items():
            # key : level(0,1,2) , items (files)
            env = lmdb.open(f"{self.data_base_path}/{self.database_store_path}/muti_env_data_{level}" , map_size= 1024 * 1024 * 1024 * 150)
            txn = env.begin(write=True)
        
            for episode , file in enumerate(tqdm(items , desc='load data')):
                # file_path  obsolutly path
                with open(file , 'rb') as f:
                    data = pickle.load(f)
                dealed_data = self.deal_data(data)
                key = f'{level}_{episode}'.encode()
                value = pickle.dumps(dealed_data)
                txn.put(key , value)
                
                if episode % 100 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            txn.commit()
            env.close()
    def fusion_data(self , raw_datas):
        
        levels_data = {
            'level0':[],
            'level1':[],
            'level2':[]
        }
        # evey path is obsolutly path like /home/getunahui/-----
        # raw_data = [raw1 ,raw2 , raw3]
        # [raw: level0 ,level1 ,level2]
        
        for raw_data in raw_datas:
            # raw_data = path
            levels = sorted(os.listdir(raw_data), key=lambda x: int(''.join(filter(str.isdigit, x))))
            # listdir raw_data = [level0 , level1 ,level2]
            # get obsolutlt path
            levels_path = [os.path.join(raw_data , i) for i in levels]
            # get evey levels files for levels_data
            for level_name , level_path in zip(levels , levels_path):
                temp = [os.path.join(level_path , i)  for i in os.listdir(level_path)]
                levels_data[level_name].extend(temp)
        return levels_data
    def get_trans(self , raw_data):
        # raw_data = [raw1 ,raw2 , raw3]
        # [raw: level0 ,level1 ,level2]
        levels_data = self.fusion_data(raw_data)
        """
        levels_data = {
            'level0':obsolutly filepath,
            'level1':obsolutly filepath,
            'level2':obsolutly filepath
        }
        """
        self.tran_to_lmdb(levels_data)
    def deal_data(self,batch_data):
        self.error = []
        seq = batch_data[0]
        current_done = batch_data[2]
        frist_state = batch_data[3]
        audio_ = self.get_data(seq,'audio')
        action = self.get_data(seq,'rl_pred')
        visual_ = self.get_data(seq,'camera')
        reward = self.get_data(seq,'reward')
        step = self.get_data(seq , 'step')


        next_audio = audio_[1:] 
        next_visual = visual_[1:]
        pre_audio  = audio_[:-1]
        pre_visual = visual_[:-1]
        done = [1 if d[0] else 0 for d in current_done]
        
        
        return pre_audio,pre_visual, next_audio, next_visual,done,reward,action
    def get_data(self , data , name):
        return torch.from_numpy(np.array(data[0][name]))
class _train_split(Dataset):
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
        self.error = list()
        
    def __getitem__(self,idx_tuple):
        # idx_tuple (path , idx)
        env_path , key = idx_tuple
        # print(env_path)
        txn = self.map(env_path)
        value = txn.get(key)
        if value is None:
            print(f"error {key}")
            self.error.append(key)
            return None 
        batch_data = pickle.loads(value)
        return self.deal_data(batch_data)
    
    def deal_data(self,batch_data):
        pre_state , next_state , done , reward , action = batch_data.values()
        # pdb.set_trace()
        return pre_state , next_state , done , reward,action
    def map(self,env_name):
        env = lmdb.open(env_name , readonly = True)
        return env.begin()
    
class _split_sampler(Sampler):
    """
    dababase [------/level0,level1,level2 | ------------]
    """
    def __init__(self, database_ , shuffle=False):
        self.databases = database_
        
        self.pattern = re.compile(r'level(\d+)')
        
        s_levels = self._s_levels(self.databases)
        
        s_keys = self._s_keys(s_levels)
        
        self.idx_data = s_keys
        
        self.shuffle = shuffle
        
        self.shuffle_s_keys()
        
        self.iter_idx = list()
    def _s_levels(self,databases):
        # match level
        # sum files path
        """
        databases like :
        ['/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/muti_env_advance_stop_encode',
        '/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/muti_env_crushed_encode'
        , '/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/muti_env_encode']
        """
        # pdb.set_trace()
        s_levels = list()
        for database in databases:
            # database [--------/~level0,level1,level2]
            levels_p_l = [os.path.join(database ,i) for i in os.listdir(database)]
            # levels_p_l [---------/level0 , ----------/level1, --------level2]
            s_levels.append(sorted(levels_p_l , key=lambda x: int(''.join(filter(str.isdigit, x)))))
            # sorted 
        return s_levels
    
    def _s_keys(self, s_levels):
        # s_levels 
        """
        [
            '-----/level0','----level1',''
            '','',''
        ]
        """
        s_keys = {
            'level0':[],
            'level1':[],
            'level2':[]
        }
        for level_tuple in zip(*s_levels):
            # level_tuple ('level0' ,'level0' , 'level0' ---------)
            for elm in level_tuple:
                # elm = '/------------/level0'
                # pdb.set_trace()
                env = lmdb.open(elm , readonly = True)
                txn = env.begin()
                cursor = txn.cursor()
                # traverse the key in txn
                for key , _ in tqdm(cursor , desc='load all key'):
                    key_str = key.decode()
                    match = self.pattern.search(key_str)
                    if match:
                        level = match.group()  # 比如 'level0_0''level0_1'
                        if level in s_keys:
                            # [(path , key)(path,key)-------]
                            s_keys[level].append((elm, key))
                        else:
                            raise ValueError(f"Unrecognized level: {level}")
                    else:
                        raise ValueError(f"Key pattern not matched: {key_str}")
                    
        return s_keys
      
    
    def shuffle_s_keys(self):

        if self.shuffle:
            for level in self.idx_data:
                random.shuffle(self.idx_data[level])
            
    def sample_data(self, sample):
        """
        sample_dict =   {
        'level_0':1, rate 0~1
        'level_1':0,
        'level_2':0,
        }
        """
        sampled_data = []

        for level, num_rate in sample.items():
            num_samples = int(len(self.idx_data[level]) * num_rate)
            if level in self.idx_data:
                if num_samples > len(self.idx_data[level]):
                    raise ValueError(f"Not enough samples in {level}: requested {num_samples}, available {len(self.idx_data[level])}")
                sampled_data.extend(self.idx_data[level][:num_samples])
            else:
                raise KeyError(f"Invalid level '{level}' not found in idx_data. Available levels: {list(self.idx_data.keys())}")
        
        if self.shuffle:
            random.shuffle(sampled_data)
        self.iter_idx = sampled_data

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.iter_idx)
        return iter(self.iter_idx)
    def __len__(self):
        return len(self.iter_idx)