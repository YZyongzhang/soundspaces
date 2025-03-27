import logging
import torch
import os
import time
import random
import ray
import habitat_sim
import pickle
from config import config
from env.v0d0 import Env
from utils.batch import *
from quaternion import from_euler_angles, as_float_array
import quaternion
# random.seed(config["random_seed"])
random.seed(int(time.time()))

class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0
        self.env = Env(config)
        self._sim = self.env._sim
        self.path_point = list()
        self.random_action = ['turn_left' , 'turn_right' ,'move_forward']
        self._idx = 0
    def reset(self):
        self._idx = 0
    def greedy_act(self, env):
        ret = list()
        self.path_point.append(env.get_agent_pos()[0])
        action = self.paths[self._idx]
        print(f"agent action: {action}")
        # logging.info(f"agent action: {action} idx :{self._idx}") 
        act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        return ret
    def rollout(self):
        self.env.reset()
        self.reset()
        self.obs = self.env._get_observations()
        self._num_episodes += 1
        done = list()
        all_r_list = list()
        env = self.env
        
        # logging.info(index)
        self.paths = self.env.get_shortest_action_list()[0]
        while True:
            rl_output_list = self.greedy_act(env)
            all_list = [self.env.step(rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            done.append(done_list)
            if all(done_list):
                for k, v in info_list[0].items():
                    logging.info(f"Env  {k}: {v}")
                break
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])
        num_success = sum([int(t) for info in info_list for t in info["success"]])

        seq_list = list()
        seq_list += env.get()

            
        result = [seq_list , {
            'path_point':self.path_point,
            'sound_pos':env.get_source_pos()[0]
        } , done ,self.obs]
        self.path_point = list()
        torch.cuda.empty_cache()
        return  result, return_, num_success

def collect():
    actor = Actor(config)
    seq_list = list()
    for num_episodes in range(100):
        t_start = time.time()
        result_lists ,_ ,_ = actor.rollout()
        path = os.path.join(f"data/RL/best_action_forward_angle_valdata",  f"rl_episode_{num_episodes}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result_lists, f)
        actor.path_point = list()
        seq_list.clear()
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"episode time {(time.time() - t_start) // 60} m {(time.time() - t_start) % 60} s")
        logging.info(f"now time is {current_time_str}")
if  __name__== "__main__":
    logging.basicConfig(filename='./data/RL/best_action_forward_angle_valdata/RLDATA.log', level=logging.INFO)
    collect()
