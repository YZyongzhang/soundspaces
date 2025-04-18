import logging
import torch
import os
import time
import random
import ray
import habitat_sim
import pickle
# from yz.config.env_config import config
from yz.config import agent_config , config
from yz.env.v0d0 import Env
from yz.utils.batch import *
random.seed(config["random_seed"])

class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0
        self.env = Env(config)
        self._sim = self.env._sim
        self.path_point = list()
        self._idx = 0
    def reset(self):
        self._idx = 0
    def greedy_act(self, env):
        ret = list()
        self.path_point.append(env.get_agent_pos()[0])
        action = self.paths[self._idx]
        print(f"agent action: {action}")
        logging.info(f"agent action: {action}") 
        act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        return ret
    def agent_random_act(self):
        ret = list()
        path_id = [0,1,2]
        act_id = random.choice(path_id)
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        return ret
    def rollout(self):
        self.env.reset()
        self.reset()
        self.obs = self.env._get_observations()
        self._num_episodes += 1
        self.random_act_num = 0
        self.random_act = True
        done = list()
        all_r_list = list()
        env = self.env
        while True:
            if self.random_act:
                rl_output_list = self.agent_random_act()
                if self.random_act_num >= 20:
                    self.random_act = False
                else:
                    self.random_act_num += 1
            else:
                self.paths = self.env.get_shortest_action_list()[0]
                rl_output_list = self.greedy_act(env)
            all_list = [self.env.step(rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            done.append(done_list)
            logging.info(f"reward is {r_list}")
            if all(done_list):
                for k, v in info_list[0].items():
                    logging.info(f"Env  {k}: {v}")
                break
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])
        num_success = sum([int(t) for info in info_list for t in info["success"]])

        seq_list = list()
        seq_list += env.get()
        
        max_step = len(done)
        
        if max_step > 80 :
            level = 2
        elif max_step > 40 and max_step < 80:
            level = 1
        else:
            level = 0
        
        result = [seq_list , {
            'path_point':self.path_point,
            'sound_pos':env.get_source_pos()[0]
        } , done ,self.obs]
        self.path_point = list()
        torch.cuda.empty_cache()
        return  result, return_, num_success,level

def collect(env_path):
    level_episode = [0,0,0]
    env_indx = 0
    for num_episodes in range(1200):
        print(num_episodes)
        if num_episodes % 100 == 0:
            level_episode = [0,0,0]
            config['scene_dir'] = env_path[env_indx]
            env_name =  env_path[env_indx][-15:-4]
            print(f"change environment name {env_name}")
            logging.info(f"this env name is ########################################{env_name}")
            os.makedirs(agent_config.BASE_PARH_COLLECT + 'muti_env_crushed_no_interference/'+ env_name + "/level0" , exist_ok=True)
            os.makedirs(agent_config.BASE_PARH_COLLECT + 'muti_env_crushed_no_interference/'+ env_name + "/level1" , exist_ok=True)
            os.makedirs(agent_config.BASE_PARH_COLLECT + 'muti_env_crushed_no_interference/'+ env_name + "/level2" , exist_ok=True)
            actor = Actor(config)
            env_indx +=1
        t_start = time.time()
        result_lists ,_ ,_,level = actor.rollout()
        level_episode[level] += 1
        path = os.path.join(f"{agent_config.BASE_PARH_COLLECT}/muti_env_crushed_no_interference/{env_name}/level{level}",  f"rl_episode_level{level}_{level_episode[level]}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result_lists, f)
        actor.path_point = list()
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"level {level} file name rl_episode_{level}_{level_episode[level]}.pkl")
        logging.info(f"episode time {(time.time() - t_start) // 60} m {(time.time() - t_start) % 60} s")
        logging.info(f"now time is {current_time_str}")
if  __name__== "__main__":

    mp3d_scene_datasets = agent_config.MP3D_SCENE_DATASET
    collect_dir = agent_config.BASE_PARH_COLLECT
    
    # exit_env_path = os.listdir('./data/RL/muti_env_data/')
    # exit_env_val = os.listdir('./data/RL/muti_env_val/')
    exit_env_advance_stop_envname = os.listdir(collect_dir + 'muti_env_crushed_no_interference/')
    # import pdb; pdb.set_trace()
    
    env_path = [os.path.join(f"{mp3d_scene_datasets}/{i}" , f"{i}.glb") for i in os.listdir(mp3d_scene_datasets) if i not in exit_env_advance_stop_envname and i != 'mp3d.scene_dataset_config.json']
    logging.basicConfig(filename= collect_dir + 'muti_env_crushed_no_interference/RLDATA.log', level=logging.INFO)
    collect(env_path)
