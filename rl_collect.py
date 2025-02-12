import logging
import torch
import os
import time
import random
import ray
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
        self._num_envs = 1
        self.envs = [Env(config) for _ in range(self._num_envs)]
        self._idx = 0
        self.num = 0
        self.action_list = ["move_forward", "turn_left", "turn_right"]
        self.path_point = [{
            'sound':[],
            'agent':[]
        }]

    def reset(self):
        self._idx = 0

    def act(self, env, env_id):
        if self.num == 0:
            self.path_point[0]['sound'].append(env.get_source_pos()[0])
        ret = list()
        num_agents = env.get_num_agents()
        for agent_id in range(num_agents):
            self.path_point[agent_id]['agent'].append(env.get_agent_pos()[0])
            logging.info(env.get_agent_pos())
            # 随机take action
            if self._idx >= len(self.paths[env_id][agent_id]):
                logging.info('here stop')
                action = "stop"
            else:
                if random.random() < self.eps:
                    action = random.sample(self.action_list, 1)[0]
                    self.paths = [env.get_shortest_action_list() for env in self.envs]
                    logging.info(self.paths)
                    self.reset()
                else:
                    action = self.paths[env_id][agent_id][self._idx]
            
            # 记录动作
            print(f"agent {agent_id} action: {action}")
            logging.info(f"agent {agent_id} action: {action} idx :{self._idx} num : {self.num}")
            act_id = env.action_str_2_id(action)

            ret.append({
                "rl_pred": act_id,
                "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
                "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
            })
        
        self._idx += 1
        self.num+=1
        return ret

    def rollout(self):
        self.reset()
        self.num = 0
        self._num_episodes += 1

        all_r_list = list()
        envs = self.envs
        config = self._config
        num_envs = self._num_envs

        input_d_list = [envs[idx].reset() for idx in range(num_envs)]
        self.paths = [env.get_shortest_action_list() for env in self.envs]
        logging.info(self.paths)

        # Generate RL training data
        while True:
            rl_output_list = [self.act(envs[env_id], env_id) for env_id in range(len(envs))]
            all_list = [env.step(rl_output) for env, rl_output in zip(self.envs, rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]

            # 记录一些信息
            all_r_list.append(r_list)

            # for i in range(num_envs):
            #     for k, v in info_list[i].items():
            #         logging.info(f"Env {i} {k}: {v}")
            logging.info(done_list)
            if all(done_list):
                for i in range(num_envs):
                    for k, v in info_list[i].items():
                        logging.info(f"Env {i} {k}: {v}")
                break

        # Test and plot return
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])
        num_success = sum([int(t) for info in info_list for t in info["success"]])

        seq_list = list()
        for env in envs:
            seq_list += env.get()

        torch.cuda.empty_cache()
        return num_envs, seq_list, return_, num_success

def collect():
    ray.init()

    num_actors = 1
    num_gpus = 1 / num_actors
    # actors = [ray.remote(num_gpus=num_gpus)(Actor).remote(config) for _ in range(num_actors)]
    actors = [Actor(config)]
    seq_list = list()  # store rl data
    for num_episodes in range(10000):
        t_start = time.time()
        logging.info(f"Episode {num_episodes}")

        # token_ids = [actor.rollout.remote() for actor in actors]
        result_list = [actor.rollout() for actor in actors]
        # result_list = ray.get(token_ids)

        for result in result_list:
            num_envs_, seq_list_, return_, num_success = result
            seq_list += seq_list_

        path = os.path.join("data/RL", f"offline_episode_RL_{num_episodes}.pkl")
        with open(path, "wb") as f:
            pickle.dump(seq_list, f)
        path_point = os.path.join('data/RL/path' ,f"path_RL_{num_episodes+1}.pkl")
        with open(path_point , 'wb') as f:
            pickle.dump(actors[0].path_point, f)
        actors[0].path_point = [{
            'sound':[],
            'agent':[]
        }]
        seq_list.clear()
        logging.info(f"offline_episode_RL_{num_episodes}.pkl")
        logging.info(f"Episode {num_episodes} seq num: {len(seq_list_)}")
        logging.info(f"Episode {num_episodes} time: {time.time()-t_start}")

if __name__ == "__main__":
    logging.basicConfig(filename='./data/RL/path/RLDATA.log', level=logging.INFO)
    collect()
