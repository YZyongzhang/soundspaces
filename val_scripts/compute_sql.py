import sys
sys.path.append('./acm')
sys.path.append('./acm/dsac')
import logging
import torch
import pdb
import os
import time
import random
import ray
import habitat_sim
import pickle
sys.path.append('/home/getuanhui/project/sound-spaces')
from yz.config import config , agent_config
from yz.env.v0d0 import Env
from yz.utils.batch import *
from yz.utils.batch import *
from yz.utils.metrics import *
from yz.net import Critic_Actor ,AVFNet
from yz.config import agent_config
random.seed(config["random_seed"])


class Actor:
    def __init__(self, config,ckpt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = f'{agent_config.EXPERIMENTS_DIR}/ckpt/2025-04-19~08-03-26/{ckpt}'
        self.avf_model_path = '/home/getuanhui/project/sound-spaces/yz/data/checkpoint/acmcheckpoint/avf_muti_env_90000.pth'
        self.agent = Critic_Actor(action_dim=4).to(self.device)
        self.agent.load_state_dict(torch.load(self.model_path))
        self.agent.eval()
        self.avf = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.avf_model_path))
        self.avf.eval()
        
        
        self._config = config
        self.env = Env(config)
        self.path_point = {
            'sound_pos':[],
            'path_point':[]
        }
    def get_action(self , visual , audio):
        visual = torch.from_numpy(visual)
        audio = torch.from_numpy(audio)
        visual = visual.unsqueeze(0)
        audio = audio.unsqueeze(0)

        visual = visual.float().to(self.device)
        audio = audio.float().to(self.device)

        combinencode = self.avf(audio,visual)
        action = self.agent(combinencode)[2].max(dim = -1)[1]
        print(action)
        return int(action)
    
    def reset(self):
        self._idx = 0
        
    def act(self):
        # if self.num == 0:
        #     self.path_point['sound_pos'].append(env.get_source_pos()[0])
        ret = list()
        self.path_point['path_point'].append(self.env.get_agent_pos()[0])
        obs = self.env._get_observations()
        visual = obs[0]['camera']
        audio = obs[0]['audio']
        action = self.get_action(visual , audio)
        ret.append({
            "rl_pred": action,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        return ret
    
    def rollout(self):
        self.env.reset()

        all_r_list = list()
        while True:
            rl_output_list = self.act()
            all_list = [self.env.step(rl_output_list)]
            # s, r, done, info
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            if all(done_list):
                for k, v in info_list[0].items():
                    logging.info(f"Env  {k}: {v}")
                break
        
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])
        success = [int(t) for info in info_list for t in info["success"]]
        shortest_distances, path_lengths = self.env.prepare_spl()

        for_spl = {
            "success": success,
            "shortest_distances": shortest_distances,
            "path_lengths": path_lengths,
        }

        seq_list = list()
        seq_list += self.env.get()

        torch.cuda.empty_cache()
        return  seq_list, return_, for_spl
def val():
    episode = 0
    model_ckpts = ['shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_320000_492.pth',
                   'shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_10000_79.pth',
                   'shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_50000_211.pth',
                   'shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_100000_286.pth',
                   'shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_150000_333.pth']
    scene_base_dir="/home/getuanhui/project/sound-spaces/data/scene_datasets/mp3d/"
    measurements = {
        'spl':[],
        'soft_spl':[],
        'success_rate':[]
    }
    envs = ['yqstnuAEVhm','YVUC4YcDtcY','Z6MFQCViBuw','ZMojNkEp431','zsNo4HB9uLZ']
    envs = agent_config.VAL_SPLIT_ENV
    seq_list = list()
    for ckpt in model_ckpts:
        for env_name in envs:
            config['scene_dir']=scene_base_dir + f'{env_name}/{env_name}.glb'
            logging.info(f"Episode {config['scene_dir']}")
            actor = Actor(config,ckpt)
            logging.info(f"Episode {env_name}")
            
            result_list = [actor.rollout()]
            
            return_list = list()
            for_spl_all = {
                "success": list(),
                "shortest_distances": list(),
                "path_lengths": list(),
            }
            
            
            for result in result_list:
                seq_list_, return_, for_spl_ = result
                seq_list += seq_list_
                return_list.append(return_)
                for_spl_all["success"] += for_spl_["success"]
                for_spl_all["shortest_distances"] += for_spl_["shortest_distances"]
                for_spl_all["path_lengths"] += for_spl_["path_lengths"]
                
            success = for_spl_all["success"]
            shortest_distances = for_spl_all["shortest_distances"]
            path_lengths = for_spl_all["path_lengths"]
            
            spl = calculate_spl(
                for_spl_all["success"],
                for_spl_all["shortest_distances"],
                for_spl_all["path_lengths"],
            )
            soft_spl = calculate_soft_spl(
                for_spl_all["shortest_distances"],
                for_spl_all["path_lengths"], 
            )
            measurements['spl'].append(spl)
            measurements['soft_spl'].append(soft_spl)
            measurements['success_rate'].append(success)
            writer.add_scalar(f'loss/spl', torch.tensor(spl), episode)
            writer.add_scalar(f'loss/softspl', torch.tensor(soft_spl), episode)
            path_point = os.path.join(f'{agent_config.EXPERIMENTS_DIR}/val' ,f"path_RL_{env_name}_{ckpt}.pkl")
            with open(path_point , 'wb') as f:
                pickle.dump([ {'spl':spl , 'soft_spl':soft_spl},actor.path_point], f)
            actor.path_point = {
                'sound_pos':[],
                'path_point':[]
            }
            for name, item in for_spl_all.items():
                writer.add_scalar(f'loss/{name}', torch.tensor(item).item(), episode)
            episode+=1
            seq_list.clear()
            logging.info(f"spl {spl} / soft_spl {soft_spl} /shortest_distances:{shortest_distances} / path_lengths {path_lengths} ,success {success}")
        
        for k, v in measurements.items():
            logging.info(f"val result : {k}: {v}")
        for k, v in measurements.items():
            logging.info(f"val result : {k}: {np.array(v).mean()}")
        
if __name__ == "__main__":
    os.makedirs(f'{agent_config.EXPERIMENTS_DIR}/val',  exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'{agent_config.EXPERIMENTS_DIR}/val/loss')
    logging.basicConfig(filename=f'{agent_config.EXPERIMENTS_DIR}/val/RLDATA.log', level=logging.INFO)
    val()
    
        
