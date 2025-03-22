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
        self.level = {
            'level1':10.0,
            'level2':30.0,
            'level3':50.0
        }
        # 将level改成角度
        self.le = self.level['level1']
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
    def angle_between_points(self,from_pos, mid_pos, to_pos):

        one_vector = mid_pos - from_pos
        two_vector = to_pos - from_pos

        dot_product = np.dot(one_vector, two_vector)

        norm_BA = np.linalg.norm(one_vector)
        norm_BC = np.linalg.norm(two_vector)

        cos_theta = dot_product / (norm_BA * norm_BC)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) 

        angle = np.degrees(theta)
        
        return angle
    def mid_point(self, agent_point_from ,agent_point_to ,r):

        while True:
            rand_pos = self._sim.pathfinder.get_random_navigable_point_near(agent_point_from , radius = r)
            if (
                np.linalg.norm(rand_pos - agent_point_from) > r - 0.5
                and np.linalg.norm(rand_pos - agent_point_from) < r + 0.5  # 圆环的空间
                and self.shortest_path(rand_pos, agent_point_from) is not None
                and self._sim.pathfinder.is_navigable(rand_pos)
                and self.angle_between_points( agent_point_from , rand_pos , agent_point_to) > 20 # 设置角度的位置
                and self.angle_between_points( agent_point_from , rand_pos , agent_point_to) < 40
            ):
                # self.paths = self.env.get_shortest_action_list(goal_pos=rand_pos)[0]
                break
            else:
                print(f'rand_pos {rand_pos} is false')
                logging.info(f'rand_pos {rand_pos} is false')
        return rand_pos    
    def greedy_act(self, env):
        ret = list()
        self.path_point.append(env.get_agent_pos()[0])
        action = self.paths[self._idx]
        print(f"agent action: {action}")
        logging.info(f"agent action: {action} idx :{self._idx}") 
        act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        return ret
    def random_act(self,env):
        # 我们这里便不要done了，全部让它跑满200个step，但是random的过程种，turn reward 便不能设置为10，这里应该是0
        ret = list()
        self.path_point.append(env.get_agent_pos()[0])
        # action = self.paths[self._idx]
        action = random.choice(self.random_action)
        print(f"agent action: {action}")
        logging.info(f"agent action: {action} idx :{self._idx}") 
        act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        
        self._idx += 1
        return ret
    def insert_list_evenly(self,list1, list2):
        result = []
        len1, len2 = len(list1), len(list2)
        
        for i in range(len2):
            result.append(list1[i])
            result.append(list2[i])
        
        result.extend(list1[len2:]) 
        return result
    def get_path_point(self):
        #
        from_pos = self.env.get_agent_pos()[0]
        to_pos = self.env.get_source_pos()[0]

        path = habitat_sim.ShortestPath()
        path.requested_start = from_pos
        path.requested_end = to_pos
        found_path = self.env._sim.pathfinder.find_path(path)
        # path.points
        # insert points
        length  = len(path.points)
        insert_points = []
        for index , value in enumerate(path.points):
            # print(index)
            if index < length - 1:
                r = 1/2 * (np.linalg.norm(path.points[index] - path.points[index+1]))
                point  = self.mid_point(path.points[index] , path.points[index+1] , r)
                print(f"插入一个元素{point}")
                logging.info(f"插入一个元素{point}")
                insert_points.append(point)
        result_point = self.insert_list_evenly(path.points , insert_points)
        logging.info(result_point)
        return result_point
        # 通过使用result_point 获取到action list
    def get_action_list(self,point):
        return self.env.get_shortest_action_list(goal_pos=point)[0]
    def rollout(self):
        self.env.reset()
        self.obs = self.env._get_observations()
        self._num_episodes += 1
        done = list()
        all_r_list = list()
        env = self.env
        self.points = self.get_path_point()
        # logging.info(f"frist path action is {self.paths}")
        for index , value in enumerate(self.points):
            logging.info(index)
            self.paths = self.env.get_shortest_action_list(goal_pos=value)[0]
            logging.info(self.paths)
            self.reset()
            while True:
                rl_output_list = self.greedy_act(env)
                logging.info(rl_output_list[0]['rl_pred'])
                if index != len(self.points) - 1 and rl_output_list[0]['rl_pred'] == 3:
                    break
                all_list = [self.env.step(rl_output_list)]
                input_d_list = [t[0] for t in all_list]  # s
                r_list = [t[1] for t in all_list]  # list of list
                done_list = [t[2] for t in all_list]
                info_list = [t[3] for t in all_list]
                all_r_list.append(r_list)
                done.append(done_list)
                logging.info(done_list)
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
        # s = f'level{(num_episodes//200) + 1 }'
        s = 'level3'
        # actor.le = actor.level[s]
        t_start = time.time()
        logging.info(f"Episode {num_episodes}")
        result_lists ,_ ,_ = actor.rollout()
        path = os.path.join(f"data/RL/forward_angle/{s}",  f"rl_episode_{s}_{num_episodes}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result_lists, f)
        actor.path_point = list()
        seq_list.clear()
        logging.info(f"offline_episode_RL_{num_episodes}.pkl")
        logging.info(f"Episode {num_episodes} time: {time.time()-t_start}")

if  __name__== "__main__":
    logging.basicConfig(filename='./data/RL/forward_angle/RLDATA.log', level=logging.INFO)
    collect()
