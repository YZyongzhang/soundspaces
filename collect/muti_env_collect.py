import logging
import torch
import os
import time
import random
import ray
import habitat_sim
import pickle
from yz.config import agent_config , config
from yz.env.v0d0 import Env
from yz.utils.batch import *
from quaternion import from_euler_angles, as_float_array
import quaternion
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
        limit = 0
        while True:
            rand_pos = self._sim.pathfinder.get_random_navigable_point_near(agent_point_from , radius = r)
            if (
                # np.linalg.norm(rand_pos - agent_point_from) > r - 0.5
                # and np.linalg.norm(rand_pos - agent_point_from) < r + 1.0  # 圆环的空间
                self.shortest_path(rand_pos, agent_point_from) is not None
                and self._sim.pathfinder.is_navigable(rand_pos)
                and self.angle_between_points( agent_point_from , rand_pos , agent_point_to) > 20 # 设置角度的位置
                and self.angle_between_points( agent_point_from , rand_pos , agent_point_to) < 40
            ):
                # self.paths = self.env.get_shortest_action_list(goal_pos=rand_pos)[0]
                break
            else:
                print(f'rand_pos {rand_pos} is false')
                if limit > 100 :
                    break
                limit += 1
                # logging.info(f'rand_pos {rand_pos} is false')
        if limit > 100 or limit == 100:
            return agent_point_from
        else:
            return rand_pos    
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
    def random_act(self):
        ret = list()
        act_id = random.choice([0,1,2])
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
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
                # logging.info(f"插入一个元素{point}")
                insert_points.append(point)
        result_point = self.insert_list_evenly(path.points , insert_points)
        # logging.info(result_point)
        return result_point
        # 通过使用result_point 获取到action list
    def get_action_list(self,point):
        return self.env.get_shortest_action_list(goal_pos=point)[0]
    def get_max_step(self , seq):
        pass
    def rollout(self):
        self.env.reset()
        self.obs = self.env._get_observations()
        self._num_episodes += 1
        done = list()
        all_r_list = list()
        env = self.env
        
        """
        进行random的调用，不断的random20step 确保能够获得碰撞数据
        """
        random_step = 0
        while random_step < 20:
            rl_output_list = self.random_act()
            all_list = [self.env.step(rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            done.append(done_list)
            logging.info(f"reward is {r_list}")
            random_step+=1
        self.points = self.get_path_point()
        # logging.info(f"frist path action is {self.paths}")
        for index , value in enumerate(self.points):
            # logging.info(index)
            self.paths = self.env.get_shortest_action_list(goal_pos=value)[0]
            # logging.info(self.paths)
            self.reset()
            while True:
                rl_output_list = self.greedy_act(env)
                # logging.info(rl_output_list[0]['rl_pred'])
                if index != len(self.points) - 1 and rl_output_list[0]['rl_pred'] == 3:
                    break
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
            cul_level = 2
        elif max_step > 40 and max_step < 80:
            cul_level = 1
        else:
            cul_level = 0
            
        result = [seq_list , {
            'path_point':self.path_point,
            'sound_pos':env.get_source_pos()[0]
        } , done ,self.obs]
        self.path_point = list()
        torch.cuda.empty_cache()
        return  result, return_, num_success , cul_level

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
            os.makedirs(agent_config.BASE_PARH_COLLECT + 'muti_env_crushed/'+ env_name + "/level0" , exist_ok=True)
            os.makedirs(agent_config.BASE_PARH_COLLECT + 'muti_env_crushed/'+ env_name + "/level1" , exist_ok=True)
            os.makedirs(agent_config.BASE_PARH_COLLECT + 'muti_env_crushed/'+ env_name + "/level2" , exist_ok=True)
            actor = Actor(config)
            env_indx +=1
        t_start = time.time()
        result_lists ,_ ,_,level = actor.rollout()
        level_episode[level] += 1
        path = os.path.join(f"{agent_config.BASE_PARH_COLLECT}/muti_env_crushed/{env_name}/level{level}",  f"rl_episode_level{level}_{level_episode[level]}.pkl")
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
    exit_env_advance_stop_envname = os.listdir(collect_dir + 'muti_env_crushed/')
    # import pdb; pdb.set_trace()
    
    env_path = [os.path.join(f"{mp3d_scene_datasets}/{i}" , f"{i}.glb") for i in os.listdir(mp3d_scene_datasets) if i not in exit_env_advance_stop_envname and i != 'mp3d.scene_dataset_config.json']
    logging.basicConfig(filename= collect_dir + 'muti_env_crushed/RLDATA.log', level=logging.INFO)
    collect(env_path)
