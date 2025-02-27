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
from new import model
random.seed(config["random_seed"])
class Actor:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = './new/rl_model.pth'
        self.agent = model().to(self.device)
        self.agent.load_state_dict(torch.load(self.model_path))
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
    def get_action(self , visual , audio):
        visual = torch.from_numpy(visual)
        audio = torch.from_numpy(audio)
        visual = visual.unsqueeze(0)
        audio = audio.unsqueeze(0)
        visual = visual.float()
        audio = audio.float()
        print(visual.dtype)
        print(audio.dtype)
        action_list  = self.agent(audio , visual)
        action = torch.max(action_list,dim=1)[1].tolist()
        print(action)
        return action[0]
        

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
            logging.info(f"agent sound source_pos is {env.get_source_pos()[0]}")
            self.path_point['sound'].append(env.get_source_pos()[0])
        ret = list()
        self.path_point['agent'].append(env.get_agent_pos()[0])
        logging.info(f"agent pos is {env.get_agent_pos()}")
        obs = env._get_observations()
        visual = obs[0]['camera']
        audio = obs[0]['audio']
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
    def mid_point(self,agent_pos = None , level = None):
        if level == None:
            level = 'level2'
        if agent_pos == None:
            agent_pos = self.env.get_agent_pos()[0]
        while True:
            rand_pos = self._sim.pathfinder.get_random_navigable_point_near(agent_pos , radius = self.level[level])
            if (
                np.linalg.norm(rand_pos - agent_pos) > 1.0
                and np.linalg.norm(rand_pos - agent_pos) < 3.0
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
    for num_episodes in range(10):
        t_start = time.time()
        logging.info(f"Episode {num_episodes}")
        result_list = [actor.rollout()]
        for result in result_list:
            seq_list_, return_, num_success = result
            seq_list += seq_list_
        # path = os.path.join("data/RL/new_random",  f"offline_episode_RL_{num_episodes}.pkl")
        # with open(path, "wb") as f:
        #     pickle.dump(seq_list, f)
        path_point = os.path.join('data/RL/val' ,f"path_RL_{num_episodes}.pkl")
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
if __name__ == "__main__":
    logging.basicConfig(filename='./data/RL/val/RLDATA.log', level=logging.INFO)
    collect()