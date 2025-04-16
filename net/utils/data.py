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
        # muti_env_data_base_path = '../../../data/database/val_database_encode'
    def tran_to_encode(self , audio , visual):
        combinencode  = self.model(audio , visual)

    def gen_database(self , database_path , level):
        encode_env = lmdb.open(f"{self.combinencode_base_path}/mutienv_data_combinencode_advance_stop_level/combinencode_level_{level}" , map_size= 1024 * 1024 * 1024 * 50)
        encode_txn = encode_env.begin(write=True)
        mydata  = USE_LMBD_DATABASE(database_path)
        keyid = 0
        for idx in tqdm(range(mydata.__len__()) , desc= 'load data'):
            # batch_data 有很多的这个数据，这次存储使用的key和value为{idx :batchdata}
            pre_audio,pre_visual, next_audio, next_visual,done,reward,action = mydata.__getitem__(idx)
            pre_state_combinencode = self.model(pre_audio , pre_visual)
            next_state_combinencode = self.model(next_audio , next_visual)
            if pre_state_combinencode.shape[0] == next_state_combinencode.shape[0]:
                for batch_idx in range(pre_state_combinencode.shape[0]):
                    key = f'level_{level}_{keyid}'.encode()
                    data = {
                        'pre_state':pre_state_combinencode[batch_idx],
                        'next_state':next_state_combinencode[batch_idx],
                        'done': torch.tensor(done[batch_idx]),
                        'reward' :reward[batch_idx], 
                        'action' :action[batch_idx],
                    }
                    # pdb.set_trace()
                    value = pickle.dumps(data)
                    encode_txn.put(key , value)
                    if keyid % 500 == 0:
                        encode_txn.commit()
                        encode_txn = encode_env.begin(write=True)
                    keyid+=1
            else:
                print('shape error!')
            # origin 8485  8485 - 27676  27676 -53934
        print(f"finally key id {keyid}")# level0 keyid = 0 - 20569 ， wen level1 key 44636  , 20570 - 65205  , level2 65205 163788
        encode_txn.commit()
        encode_env.close()
            
class TRANS_TO_DATABASE_FROM_RAWDATA:
    def __init__(self):
        self.data_base_path = agent_config.RELATIVE_DATABASE_DIR
    def tran_to_lmdb(self , files , level):
        # pdb.set_trace()
        env = lmdb.open(f"{self.data_base_path}/muti_env_advance_stop/muti_env_data_{level}" , map_size= 1024 * 1024 * 1024 * 100)
        txn = env.begin(write=True)
        
        for episode , file in enumerate(tqdm(files , desc='load data')):
            # print(file)
            with open(file , 'rb') as f:
                data = pickle.load(f)
            key = f'episode_{episode}'.encode()
            value = pickle.dumps(data)
            txn.put(key , value)
            
            if episode % 100 == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        env.close()

    def muti_data(self , path):
        result0 = list()
        result1 = list()
        result2 = list()
        thispath = [f'{path}/level0',f'{path}/level1',f'{path}/level2']
        files0 = [os.path.join(thispath[0] , i) for i in os.listdir(thispath[0])]
        random.shuffle(files0)
        files1 = [os.path.join(thispath[1] , i) for i in os.listdir(thispath[1])]
        random.shuffle(files1)
        files2 = [os.path.join(thispath[2] , i) for i in os.listdir(thispath[2])]
        random.shuffle(files2)
        
        result0.extend(files0)
        result1.extend(files1)
        result2.extend(files2)
        return result0 , result1 , result2
    def fusion_data(self):
        # path = [f'{agent_config.BASE_PARH_COLLECT}/success_and_stop_data/level0' ,f'{agent_config.BASE_PARH_COLLECT}/success_and_stop_data/level1', f'{agent_config.BASE_PARH_COLLECT}/success_and_stop_data/level2']
        # data_file = list()
        # files0 = [os.path.join(path[0] , i) for i in os.listdir(path[0])]
        # files1 = [os.path.join(path[1] , i) for i in os.listdir(path[1])]
        # files2 = [os.path.join(path[2] , i) for i in os.listdir(path[2])]
        # data_file.extend(files0)
        # data_file.extend(files1)
        # data_file.extend(files2)
        
        # muti_env = f'{agent_config.BASE_PARH_COLLECT}/muti_env_data'
        # envs_data = [os.path.join(muti_env , i) for i in os.listdir(muti_env) if i != 'RLDATA.log']
        # muti_env_data_level0 = list()
        # muti_env_data_level1 = list()
        # muti_env_data_level2 = list()

        # for i in envs_data:
        #     level0 , level1 , level2 = self.muti_data(i)
        #     muti_env_data_level0.extend(level0)
        #     muti_env_data_level1.extend(level1)
        #     muti_env_data_level2.extend(level2)

        # result0 = files0 + muti_env_data_level0
        # result1 = files1 + muti_env_data_level1
        # result2 = files2 + muti_env_data_level2
        # return result0 , result1 , result2
        muti_env = f'{agent_config.BASE_PARH_COLLECT}/muti_env_advance_stop'
        envs_data = [os.path.join(muti_env , i) for i in os.listdir(muti_env) if i != 'RLDATA.log']
        muti_env_data_level0 = list()
        muti_env_data_level1 = list()
        muti_env_data_level2 = list()

        for i in envs_data:
            level0 , level1 , level2 = self.muti_data(i)
            muti_env_data_level0.extend(level0)
            muti_env_data_level1.extend(level1)
            muti_env_data_level2.extend(level2)
        return muti_env_data_level0 , muti_env_data_level1 , muti_env_data_level2
    def get_trans(self):
        result0 , result1 ,result2 = self.fusion_data()
        print('en1')
        self.tran_to_lmdb(result0 , 0)
        self.tran_to_lmdb(result1 , 1)
        self.tran_to_lmdb(result2 , 2)
        
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
    def __getitem__(self,idx):
        # pdb.set_trace()
        # print(self.files[idx])
        key = f'episode_{idx}'.encode()
        value = self.txn.get(key)
        batch_data = pickle.loads(value)
        return self.deal_data(batch_data)
    
            
    def get_data(self , data , name):
        d = list()
        for i in range(len(data)):
            d.extend(self.totensor(data[i][name]))
        return torch.stack(d)
    def totensor(self,data):
        temp = torch.from_numpy(data).to(self.device)
        return temp[1:]
    def get_error(self):
        return self.error
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
        # reward = reward_[:index]
        # action = action_[:index]

        # next_audio = audio_[1:index+1] 
        # next_visual = visual_[1:index+1]
        # pre_audio  = audio_[:index]
        # pre_visual = visual_[:index]
        # done = [1 if d[0] else 0 for d in current_done]
        # 收集数据得时候发现最后一个step多了出来，所以这里我就之间去除最后一个数据
        reward = reward_[:index-1]
        action = action_[:index-1]

        next_audio = audio_[1:index] 
        next_visual = visual_[1:index]
        pre_audio  = audio_[:index-1]
        pre_visual = visual_[:index-1]
        done = [1 if d[0] else 0 for d in current_done[:-1]]
        
        # print(reward)
        # print(action)
        # print(done)
        # time.sleep(2)
        # 输出表示没有问题
        return pre_audio,pre_visual, next_audio, next_visual,done,reward,action