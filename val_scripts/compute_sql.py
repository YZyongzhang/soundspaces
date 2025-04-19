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
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = f'{agent_config.EXPERIMENTS_DIR}/ckpt/2025-04-19~08-03-26/shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_320000_492.pth'
        self.avf_model_path = '/home/getuanhui/project/sound-spaces/yz/data/checkpoint/acmcheckpoint/avf_muti_env_90000.pth'
        self.agent = Critic_Actor(action_dim=4).to(self.device)
        self.agent.load_state_dict(torch.load(self.model_path))
        self.agent.eval()
        self.avf = AVFNet(hid_dim=128 , out_put=4 ,width_dim=128 , height_dim=36).to(self.device)
        self.avf.load_state_dict(torch.load(self.avf_model_path))
        self.avf.eval()
        
        
        self._config = config
        self._num_episodes = 0
        self.env = Env(config)
        self._sim = self.env._sim
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
        
    def act(self, env):
        if self.num == 0:
            self.path_point['sound_pos'].append(env.get_source_pos()[0])
        ret = list()
        self.path_point['path_point'].append(env.get_agent_pos()[0])
        obs = env._get_observations()
        visual = obs[0]['camera']
        audio = obs[0]['audio']
        # pdb.set_trace()
        action = self.get_action(visual , audio)
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
        self.paths = list()
        env.reset()
        while True:
            rl_output_list = self.act(env)
            all_list = [self.env.step(rl_output_list)]
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
        num_success = sum([int(t) for info in info_list for t in info["success"]])
        success_list = [int(t) for info in info_list for t in info["success"]]
        last_geo_distance = sum(
            [t for info in info_list for t in info["geodesic_distance"]]
        )
        shortest_dis_list = list()
        path_list = list()
        shortest_distances, path_lengths = env.prepare_spl()
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
def val():
    actor = Actor(config)
    measurements = {
        'spl':[],
        'soft_spl':[],
        'success_rate':[]
    }
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
        measurements['spl'].append(spl)
        measurements['soft_spl'].append(soft_spl)
        measurements['success_rate'].append(success)
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
        logging.info(f"Episode {num_episodes} time: {time.time()-t_start}")
    
    for k, v in measurements.items():
        logging.info(f"val result : {k}: {v}")
    for k, v in measurements.items():
        logging.info(f"val result : {k}: {np.array(v).mean()}")
if __name__ == "__main__":
    os.makedirs(f'{agent_config.EXPERIMENTS_DIR}/val',  exist_ok=True)
    logging.basicConfig(filename=f'{agent_config.EXPERIMENTS_DIR}/val/RLDATA.log', level=logging.INFO)
    val()
    
        
