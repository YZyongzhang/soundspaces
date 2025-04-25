import logging
import torch
import os
import time
import random
import ray
import habitat_sim
from habitat_sim.utils.common import quat_to_magnum
import quaternion as qt
import pickle
from yz.config import agent_config , config
from yz.env.v0d0 import Env
from yz.utils.batch import *
from quaternion import from_euler_angles, as_float_array
import quaternion
import gc
random.seed(int(time.time()))

class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0
        self.env = Env(config)
        self._sim = self.env._sim
        self.path_point = list()
    def reset(self):
        self._idx = 0
    
    def check_greedflower_error(self, state_position , goal_pos,env):
        # raise in habitat-smi/nav/greedflower.findpath,if len(path) = 0,this function avoid it
        # self._greedy_follower [agentid, greedyfollower]
        # state.rotation   qt.quaternion(1, 0, 0, 0)
        defualt_rotation = qt.quaternion(1, 0, 0, 0)
        path = [env.impl[agent_id].find_path(
            quat_to_magnum(defualt_rotation), state_position, goal_pos
        )
        for agent_id in range(1)]
        
        s = [False if len(path[i]) == 0 else True for i in range(1)]
        
        if all(s):
            return True
        return False
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
    def noised_points(self, agent_point_from ,agent_point_to ,r ,env):
        # avoid all sample point ,so add a limit if reached the limit ,while break , retrun []
        limit = 0
        while True:
            
            rand_pos = env._sim.pathfinder.get_random_navigable_point_near(agent_point_from , radius = r)
            if (
                (env._sim.pathfinder.is_navigable(rand_pos))
                and self.check_greedflower_error(rand_pos , agent_point_to,env)
                and self.check_greedflower_error(agent_point_from, rand_pos,env)
                and self.angle_between_points( agent_point_from , rand_pos , agent_point_to) > 20 # 设置角度的位置
                and self.angle_between_points( agent_point_from , rand_pos , agent_point_to) < 40
            ):
                break
            else:
                print(f'rand_pos {rand_pos} is false')
                limit += 1
                if limit > 100 :
                    break
                
        if limit > 100:
            return []
        else:
            return [rand_pos]  
         
    def greedy_act(self, env):
        ret = list()
        if_mid_reached_goal = False
        self.path_point.append(env.get_agent_pos()[0])
        action = self.paths[self._idx]
        print(f"agent action: {action}")
        # import pdb; pdb.set_trace()
        logging.info(f"agent action: {action}") 
        act_id = env.action_str_2_id(action) 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        if self._idx == len(self.paths) - 1 and act_id != 3:
            if_mid_reached_goal = True
        self._idx += 1
        return ret , if_mid_reached_goal
    def random_act(self):
        ret = list()
        act_id = random.choice([0,0,0,1,2])
        logging.info(f"agent action: {act_id}") 
        ret.append({
            "rl_pred": act_id,
            "lstm_h": np.zeros((self._config["hid_dim_l"],), np.float32),
            "lstm_c": np.zeros((self._config["hid_dim_l"],), np.float32),
        })
        return ret
    def merge_two_list(self,list1, list2):
        # list1 [elm,elm]
        # list2 [[],[]]
        # len(list1) - 1 = len(list2)
        result = []
        len2 =  len(list2)
        # import pdb; pdb.set_trace()
        # [elm_1 , elm_2 , elm_1 , elm2]
        for i in range(len2):
            result.append(list1[i])
            if list2[i]:  # list2[i] not null
                result.append(list2[i][0])
        
        result.extend(list1[len2:])
        return result
    def get_path_point(self,env):
        # get path points list
        from_pos = env.get_agent_pos()[0]
        to_pos = env.get_source_pos()[0]

        path = habitat_sim.ShortestPath()
        path.requested_start = from_pos
        path.requested_end = to_pos
        found_path = self.env._sim.pathfinder.find_path(path)
        # path.points
        return path.points
    def insert_noised(self, mid_path_list,env):
        
        insert_points = []
        for index , value in enumerate(mid_path_list[:-1]):
            # (index , index +1)
            # get the noise point length r
            r = np.linalg.norm(mid_path_list[index] - mid_path_list[index+1])
            # get the noised points
            point  = self.noised_points(mid_path_list[index] , mid_path_list[index+1] , r ,env)
            # store the point
            insert_points.append(point)
        # insert_point = [[p],[p],[p],[],[],[p]] ,may be like that
        result_point = self.merge_two_list(mid_path_list , insert_points)
        # result_point = [elm,elm,elm,elm]
        return result_point
    
    def get_paths(self,env):
        ## get mid point list
        mid_path_list = self.get_path_point(env)
        ## insert noise into mid_path_list
        noised_path_list = self.insert_noised(mid_path_list,env)
        ## noised_path_list = [elm,elm,elm]
        ## finnally path_string like turn_left list
        # finnally_path = self.get_action_list(noised_path_list,env)
        
        return noised_path_list
        
    def noise_greedy(self):
        # in this self.env.reset() don't reset the self._data in env function .so 
        # when we use self.env.get() the environment return all of data ,include last episode and far long befor
        # this is a serise bug of memory leak. to fix this, we can use a new temp var env = self.env and use env to reset()
        env = self.env
        env.reset()
        self.obs = env._get_observations()
        self._num_episodes += 1
        done = list()
        all_r_list = list()
        points = self.get_paths(env)
        done_list = [False]
        # import pdb; pdb.set_trace()
        for idx , point in enumerate(points[1:]):
            if all(done_list):
                # double check all done, if env done, all cycle exit
                break
            self.reset()
            if idx == len(points) - 2:
                self.paths = env.get_shortest_action_list(goal_pos=point)[0]
            else:
                self.paths = env.get_shortest_action_list(goal_pos=point)[0][:-1]
            while True:
                if not self.paths:
                    break
                rl_output_list, if_mid_reached_goal = self.greedy_act(env)
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
                if if_mid_reached_goal:
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
        seq_list.clear()
        return  result, return_, num_success , cul_level
   
    def noise_greedy_advance_stop(self):
        self.env.reset()
        self.obs = self.env._get_observations()
        self._num_episodes += 1
        done = list()
        all_r_list = list()
        env = self.env
        points = self.get_paths(env)
        done_list = [False]
        # import pdb; pdb.set_trace()
        for idx , point in enumerate(points[1:]):
            if all(done_list):
                # double check all done, if env done, all cycle exit
                break
            self.reset()
            if idx == len(points) - 2:
                # remove five step before stop to avachieve the advance stop
                self.paths = env.get_shortest_action_list(goal_pos=point)[0]
                self.paths = self.paths[:-5] if len(self.paths) > 5 else []
                self.paths.extend(['stop'])
            else:
                self.paths = env.get_shortest_action_list(goal_pos=point)[0][:-1]
            while True:
                if not self.paths:
                    break
                rl_output_list, if_mid_reached_goal = self.greedy_act(env)
                all_list = [self.env.step(rl_output_list)]
                input_d_list = [t[0] for t in all_list]  # s
                r_list = [t[1] for t in all_list]  # list of list
                done_list = [t[2] for t in all_list]
                info_list = [t[3] for t in all_list]
                all_r_list.append(r_list)
                done.append(done_list)
                logging.info(f"reward is {r_list}")
                
                """
                this will have a bug about max step 
                if env.step return done = true , it also will be environment got the
                max step limit ,so return done = true.
                in this case , if_mid_reached_goal befor done_list to check, and if
                if_mid_reached_goal = true . program will ingore the done_list condition
                that will be occerr a bug that env is end, but we still input step to environment
                
                so in this i change the two position, let the done_list in above
                """
                if all(done_list):
                    for k, v in info_list[0].items():
                        logging.info(f"Env  {k}: {v}")
                    break
                if if_mid_reached_goal:
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
    
    def noise_greedy_crashed(self):
        env = self.env
        env.reset()
        self.obs = env._get_observations()
        self._num_episodes += 1
        done = list()
        all_r_list = list()
        # action random 20step
        for i in range(20):
            rl_output_list = self.random_act()
            all_list = [env.step(rl_output_list)]
            input_d_list = [t[0] for t in all_list]  # s
            r_list = [t[1] for t in all_list]  # list of list
            done_list = [t[2] for t in all_list]
            info_list = [t[3] for t in all_list]
            all_r_list.append(r_list)
            done.append(done_list)
            logging.info(f"reward is {r_list}")
        # get noise pathpoint to greedy nav
        points = self.get_paths(env)
        done_list = [False]
        # import pdb; pdb.set_trace()
        for idx , point in enumerate(points[1:]):
            if all(done_list):
                # double check all done, if env done, all cycle exit
                break
            self.reset()
            if idx == len(points) - 2:
                # remove five step before stop to avachieve the advance stop
                self.paths = env.get_shortest_action_list(goal_pos=point)[0]
                self.paths = self.paths[:-5] if len(self.paths) > 5 else []
                self.paths.extend(['stop'])
            else:
                self.paths = env.get_shortest_action_list(goal_pos=point)[0][:-1]
            while True:
                if not self.paths:
                    break
                rl_output_list, if_mid_reached_goal = self.greedy_act(env)
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
                if if_mid_reached_goal:
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
    
    def greedy(self):
            self.env.reset()
            self.obs = self.env._get_observations()
            self._num_episodes += 1
            done = list()
            all_r_list = list()
            env = self.env
            # self.paths = self.get_paths(env)
            self.paths = env.get_shortest_action_list()[0]
            self.reset()
            while True:
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

def collect(env_path , collect_name):
    level_episode = [0,0,0]
    for env in env_path:
        level_episode = [0,0,0]
        config['scene_dir'] = env
        print(f"change environment name {env}")
        logging.info(f"this env name is ########################################{env}")
        os.makedirs(agent_config.BASE_PARH_COLLECT + f'{collect_name}/'+ env + "/level0" , exist_ok=True)
        os.makedirs(agent_config.BASE_PARH_COLLECT + f'{collect_name}/'+ env + "/level1" , exist_ok=True)
        os.makedirs(agent_config.BASE_PARH_COLLECT + f'{collect_name}/'+ env + "/level2" , exist_ok=True)
        actor = Actor(config)
        for num_episodes in range(10):
            print(num_episodes)
            t_start = time.time()
            # result_lists ,_ ,_,level = actor.noise_greedy_crashed()
            result_lists ,_ ,_,level = actor.noise_greedy_advance_stop()
            # result_lists ,_ ,_,level = actor.noise_greedy()
            level_episode[level] += 1
            path = os.path.join(f"{agent_config.BASE_PARH_COLLECT}/{collect_name}/{env}/level{level}",  f"rl_episode_level{level}_{level_episode[level]}.pkl")
            with open(path, "wb") as f:
                pickle.dump(result_lists, f)
            actor.path_point = list()
            current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.info(f"level {level} file name rl_episode_{level}_{level_episode[level]}.pkl")
            logging.info(f"episode time {(time.time() - t_start) // 60} m {(time.time() - t_start) % 60} s")
            logging.info(f"now time is {current_time_str}")
        del actor
        torch.cuda.empty_cache()
        gc.collect()
if  __name__== "__main__":
    check_exit_env = True
    mp3d_scene_datasets = agent_config.MP3D_SCENE_DATASET
    collect_dir = agent_config.BASE_PARH_COLLECT
    train_env_split = agent_config.ENV_SPLIT['train']
    val_env_split = agent_config.ENV_SPLIT['val']
    test_env_split = agent_config.ENV_SPLIT['test']
    collect_name = 'noise_advance_stop_train_split'
    # collect_name = 'noise_greedy_crashed'
    # collect_name = 'noise_train_split'
    os.makedirs(collect_dir + f'{collect_name}/', exist_ok=True)
    if check_exit_env:
        exit_envname = os.listdir(collect_dir + f'{collect_name}')
        train_env_split = [i for i in train_env_split if i not in exit_envname]
    logging.basicConfig(filename= collect_dir + f'{collect_name}/RLDATA.log', level=logging.INFO)
    collect(train_env_split , collect_name)
