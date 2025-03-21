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
random.seed(config["random_seed"])

class Actor:
    def __init__(self, config):
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
            'sound':[],
            'agent':[]
        }
        self.level = {
            'level1':2.0,
            'level2':5.0,
            'level3':10.0
        }
        self.le = self.level['level1']
        self.if_get_random_point = True

    def reset(self):
        self._idx = 0
        
    def shortest_path(self, from_pos, to_pos):
        """
        Depreciated, using built-in shortestpath method, granularity is not enough
        """
        path = habitat_sim.ShortestPath()
        path.requested_start = from_pos
        path.requested_end = to_pos
        found_path = self._sim.pathfinder.find_path(path)
        path_results = (found_path, path.geodesic_distance, path.points)
        if len(path_results[-1]) > 1:
            return path_results[-1][1]
        else:
            return None
    def act(self, env):
        if self.num == 0:
            # logging.info(f"agent sound source_pos is {env.get_source_pos()[0]}")
            self.path_point['sound'].append(env.get_source_pos()[0])
        ret = list()
        self.path_point['agent'].append(env.get_agent_pos()[0])
        # logging.info(f"agent pos is {env.get_agent_pos()}")
        if self._idx % 10 == 0  and self.if_get_random_point and self._idx != 0: # 每五步进行一次随机点选取
            self.if_get_random_point = False
            self.path_id+=1
            self.mid_point() # 找到一个随机点
            self.reset() # 重置self._idx = 0 ， 由于path_id不是0，因此之后不执行重新选点
            # logging.info(self.paths)
        action = self.paths[self._idx]
        if action == "stop" and self.path_id != 10:
            self.paths = self.env.get_shortest_action_list(goal_pos=self.env.get_source_pos()[0])[0]
            self.reset()
            action = self.paths[self._idx]
            self.if_get_random_point = True
            # 记录动作
        if action == "stop" and self.path_id == 10:
            self.paths = self.env.get_shortest_action_list(goal_pos=self.env.get_source_pos()[0])[0]
            self.reset()
            action = self.paths[self._idx]
        print(f"agent action: {action}")
        logging.info(f"agent action: {action} idx :{self._idx} num : {self.num} pathid {self.path_id}") 
        act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        self.num+=1
        return ret
    def mid_point(self,agent_pos = None , level = None):
        if level == None:
            level = self.le
        if agent_pos == None:
            agent_pos = self.env.get_agent_pos()[0]
        while True:
            rand_pos = self._sim.pathfinder.get_random_navigable_point_near(agent_pos , radius = self.le)
            if (
                np.linalg.norm(rand_pos - agent_pos) > 1.0
                and np.linalg.norm(rand_pos - agent_pos) < 10.0
                and self.shortest_path(rand_pos, agent_pos) is not None
                and self._sim.pathfinder.is_navigable(rand_pos)
            ):
                self.paths = self.env.get_shortest_action_list(goal_pos=rand_pos)[0]
                break
            else:
                print(f'rand_pos {rand_pos} is false')
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
        self.paths = self.env.get_shortest_action_list()[0]
        logging.info(f"frist path action is {self.paths}")
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

        seq_list = list()
        seq_list += env.get()

        torch.cuda.empty_cache()
        return  seq_list, return_, num_success

def collect():
    actor = Actor(config)
    seq_list = list()
    for num_episodes in range(600):
        s = f'level{(num_episodes//200) + 1 }'
        actor.le = actor.level[s]
        t_start = time.time()
        logging.info(f"Episode {num_episodes}")
        result_list = [actor.rollout()]
        for result in result_list:
            seq_list_, return_, num_success = result
            seq_list += seq_list_
        path = os.path.join(f"data/RL/level_rate/{s}/level",  f"{s}_episode_{num_episodes}.pkl")
        with open(path, "wb") as f:
            pickle.dump(seq_list, f)
        path_point = os.path.join(f"data/RL/level_rate/{s}/path" ,f"{s}_eposode_path_{num_episodes}.pkl")
        with open(path_point , 'wb') as f:
            pickle.dump(actor.path_point, f)
        actor.path_point = {
            'sound':[],
            'agent':[]
        }
        seq_list.clear()
        logging.info(f"offline_episode_RL_{num_episodes}.pkl")
        logging.info(f"Episode {num_episodes} seq num: {len(seq_list_)}")
        logging.info(f"Episode {num_episodes} time: {time.time()-t_start}")

if  __name__== "__main__":
    logging.basicConfig(filename='./data/RL/level/RLDATA.log', level=logging.INFO)
    collect()
