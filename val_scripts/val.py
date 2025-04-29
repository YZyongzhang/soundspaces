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
from yz.config import config , agent_config
from yz.env.v0d0 import Env
from yz.utils.batch import *
from yz.utils.batch import *
from yz.utils.metrics import *
from yz.net import AVFNet ,Critic_Actor
from yz.config import agent_config
random.seed(config["random_seed"])
class Actor:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = f'{agent_config.EXPERIMENTS_DIR}/ckpt/2025-04-28~22-07-16/shuffle_mutienv_cql_dn_combinencode_level_0_and_1_2.pth'
        self.avf_model_path = '/home/getuanhui/project/sound-spaces/yz/data/checkpoint/acmcheckpoint/avf_muti_env_90000.pth'
        self.agent = Critic_Actor(state_size=64,action_size=4,gru_inputsize=128,gru_hidden_size=64,device=self.device)
        self.agent.load_state_dict(torch.load(self.model_path))
        self.agent.eval()
        self.avf = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.avf_model_path))
        self.avf.eval()
        self._config = config
        self._num_episodes = 0
        self.eps = 0.1
        self.env = Env(config)
        self._idx = 0 # 遍历paths
        self.num = 0 # 维护每一个episode步数
        self.path_id = 0 # 随机点次数
        self.action_list = ["move_forward", "move_forward","turn_left", "turn_right"]
        self._sim = self.env._sim
        self.path_point = {
            'sound_pos':[],
            'path_point':[]
        }
        self.hidden = None
    def get_action(self , visual , audio):
        # pdb.set_trace()
        visual = torch.from_numpy(visual)
        audio = torch.from_numpy(audio)
        visual = visual.unsqueeze(0)
        audio = audio.unsqueeze(0)

        visual = visual.float().to(self.device)
        audio = audio.float().to(self.device)

        combinencode = self.avf(audio,visual)
        # avf model
        if self.hidden == None:
            combinencode,h = self.agent.gru(combinencode)
            self.hidden == h
        else:
            combinencode,h = self.agent.gru(combinencode,self.hidden)
            self.hidden = h
        pdb.set_trace()
        action = self.agent.critic2(combinencode).max(dim = -1)[1]
        # pdb.set_trace()
        if action == 3:
            self.hidden = None
        print(action)
        return int(action)
    
    def reset(self):
        self._idx = 0
        
    def act(self, env):
        if self.num == 0:
            logging.info(f"agent sound source_pos is {env.get_source_pos()[0]}")
            self.path_point['sound_pos'].append(env.get_source_pos()[0])
        ret = list()
        self.path_point['path_point'].append(env.get_agent_pos()[0])
        logging.info(f"agent pos is {env.get_agent_pos()}")
        obs = env._get_observations()
        visual = obs[0]['camera']
        audio = obs[0]['audio']
        # pdb.set_trace()
        action = self.get_action(visual , audio)
        print(f"agent action: {action}")
        logging.info(f"agent action: {action} idx :{self._idx} num : {self.num} pathid {self.path_id}") 
        # act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": action,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        self.num+=1
        return ret
    
    def rollout(self):
        self.reset()
        self.num = 0
        self.path_id = 0
        self._num_episodes += 1

        all_r_list = list()
        env = self.env
        config = self._config
        self.paths = list()
        input_d_list = [env.reset()]
        while True:
            rl_output_list = self.act(env)
            all_list = [self.env.step(rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            logging.info(done_list)
            if all(done_list):
                for k, v in info_list[0].items():
                    logging.info(f"Env  {k}: {v}")
                break
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])
        num_success = sum([int(t) for info in info_list for t in info["success"]])
        success_list = [int(t) for info in info_list for t in info["success"]]
        last_geo_distance = sum(
            [t for info in info_list for t in info["geodesic_distance"]]
        )
        shortest_dis_list = list()
        path_list = list()
        shortest_distances, path_lengths = env.prepare_spl()
        logging.info(f'{shortest_distances, path_lengths}')
        shortest_dis_list += shortest_distances
        path_list += path_lengths

        for_spl = {
            "success": success_list,
            "shortest_distances": shortest_distances,
            "path_lengths": path_lengths,
        }

        seq_list = list()
        seq_list += env.get()

        torch.cuda.empty_cache()
        return  seq_list, return_, sum(success_list), last_geo_distance, for_spl
def collect():
    actor = Actor(config)
    seq_list = list()
    for num_episodes in range(30):
        t_start = time.time()
        logging.info(f"Episode {num_episodes}")
        result_list = [actor.rollout()]
        
        return_list = list()
        num_success_list = list()
        last_geo_distance_list = list()
        for_spl_all = {
            "success": list(),
            "shortest_distances": list(),
            "path_lengths": list(),
        }
        
        
        for result in result_list:
            seq_list_, return_, num_success, last_geo_distance_, for_spl_ = result
            seq_list += seq_list_
            return_list.append(return_)
            num_success_list.append(num_success)
            last_geo_distance_list.append(last_geo_distance_)
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
        
        path_point = os.path.join(f'{agent_config.EXPERIMENTS_DIR}/val' ,f"path_RL_{num_episodes}.pkl")
        with open(path_point , 'wb') as f:
            pickle.dump([ {'spl':spl , 'soft_spl':soft_spl},actor.path_point], f)
        actor.path_point = {
            'sound_pos':[],
            'path_point':[]
        }
        seq_list.clear()
        logging.info(f"offline_episode_RL_{num_episodes}.pkl")
        logging.info(f"spl {spl} / soft_spl {soft_spl} /shortest_distances:{shortest_distances} / path_lengths {path_lengths} ,success {success}")
        logging.info(f"Episode {num_episodes} seq num: {len(seq_list_)}")
        logging.info(f"Episode {num_episodes} time: {time.time()-t_start}")
if __name__ == "__main__":
    logging.basicConfig(filename=f'{agent_config.EXPERIMENTS_DIR}/val/RLDATA.log', level=logging.INFO)
    collect()
    
        
