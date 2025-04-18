import sys
import logging
import torch
import pdb
import os
import time
import random
import ray
import habitat_sim
import pickle
from yz.config import config
# from yz.config.env_config import config
from yz.env.v0d0 import Env
from yz.utils.batch import *
from yz.utils.batch import *
from yz.utils.metrics import *
from yz.net import AVFNet ,AVNet
random.seed(config["random_seed"])

@ray.remote
class Actor:
    def __init__(self, model):
        # self._config = config
        # self.model = model
        # self._num_episodes = 0
        # self.env = Env(config)
        # self._sim = self.env._sim
        self._config = config
        self._num_episodes = 0

        self._num_envs = config["num_envs_per_actor"]
        self.envs = [Env(config) for _ in range(self._num_envs)]

        # self._eps = config["best_eps"]

        self._idx = 0
    def get_action(self , visual , audio):
        visual = torch.from_numpy(visual)
        audio = torch.from_numpy(audio)
        visual = visual.unsqueeze(0)
        audio = audio.unsqueeze(0)

        visual = visual.float().to(self.device)
        audio = audio.float().to(self.device)

        combinencode = self.avf(audio,visual)
        action = self.agent(combinencode)[1].max(dim = -1)[1]
        print(action)
        return int(action)
    
    def reset(self):
        self._idx = 0
        
    def act(self, env):
        ret = list()
        obs = env._get_observations()
        visual = obs[0]['camera']
        audio = obs[0]['audio']
        action = self.get_action(visual , audio)
        ret.append({
            "rl_pred": action,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        return ret
    
    def rollout(self):
        self.reset()
        self.path_id = 0
        self._num_episodes += 1

        all_r_list = list()
        env = self.env
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
    def val(self):
        actors = self.model
        measurements = {
            'spl':[],
            'soft_spl':[],
            'success_rate':[],
            'sum_reward':[]
        }
        
        for num_episodes in range(10):
            
            # pdb.set_trace()
            result_list = self.rollout()
            # pdb.set_trace()
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
            measurements['sum_reward'].append(result_list)
        return measurements  
@ray.remote
class mmActor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0

        self._num_envs = config["num_envs_per_actor"]
        self.envs = [Env(config) for _ in range(self._num_envs)]

        # self._eps = config["best_eps"]

        self._idx = 0

    def reset(self):
        self._idx = 0

    def act(self, env, env_id):
        # at probability eps, take random action
        # otherwise, take best action
        ret = list()
        num_agents = env.get_num_agents()
        for agent_id in range(num_agents):
            if self._idx >= len(self.paths[env_id][agent_id]):
                action = "stop"
            else:
                action = self.paths[env_id][agent_id][self._idx]
            print("agent", agent_id, action)

            act_id = env.action_str_2_id(action)

            ret.append(
                {
                    "rl_pred": act_id,
                    "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
                    "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
                }
            )
        self._idx += 1

        return ret

    def rollout(self):
        self.reset()

        self._num_episodes += 1

        all_r_list = list()

        envs = self.envs
        config = self._config
        num_envs = self._num_envs

        input_d_list = [envs[idx].reset() for idx in range(num_envs)]

        self.paths = [env.get_shortest_action_list() for env in self.envs]

        # Generate RL training data
        while True:
            rl_output_list = [self.act(envs[env_id], env_id) for env_id in range(len(envs))]

            all_list = [
                env.step(rl_output) for env, rl_output in zip(envs, rl_output_list)
            ]

            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]

            # Record some info
            all_r_list.append(r_list)

            for i in range(num_envs):
                for k, v in info_list[i].items():
                    logger.info(f"Env {i} {k}: {v}")

            if all(done_list):
                break

        # Test and plot return
        return_ = sum([sum([sum(r) for r in r_list]) for r_list in all_r_list])

        num_success = sum([int(t) for info in info_list for t in info["success"]])

        seq_list = list()
        for env in envs:
            seq_list += env.get()

        torch.cuda.empty_cache()

        return num_envs, seq_list, return_, num_success

# if __name__ == "__main__":
#     # pdb.set_trace()
#     val()
    
        
