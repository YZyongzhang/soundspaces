from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box, Dict
import habitat_sim.sim
import soundspaces as ss
import numpy as np
import quaternion
import copy

from utils.angles import *
from quaternion import from_euler_angles, as_float_array
from SAM import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator

# from utils.audio import ChunkedAudio
from scipy.io.wavfile import write
from scipy.signal import fftconvolve
import librosa

from utils.time import time_count
import time

import math
from collections import deque
from queue import PriorityQueue

from typing import List

EPS = 1e-3

import attr


@attr.s(auto_attribs=True, slots=True)
class StopSpec:
    pass


@habitat_sim.registry.register_move_fn(body_action=True)
class Stop(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: StopSpec):
        pass


class MultiAudioEnv(ParallelEnv):
    metadata = {"name": "multi_audio_nav"}

    id2str = {
        0: "move_forward",
        1: "turn_left",
        2: "turn_right",
        3: "stop",
    }

    str2id = {
        "move_forward": 0,
        "turn_left": 1,
        "turn_right": 2,
        "stop": 3,
    }

    def __init__(self, config: dict):
        self.ckp = '/home/getuanhui/project/sound-spaces/yz/SAM/checkpoint/sam_vit_b_01ec64.pth'
        self.sam = sam_model_registry['vit_b'](self.ckp).to("cuda:0")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        # deep copy config
        self._config = config.copy()
        self._num_agents = config["agents_num"]
        self._num_sources = config["sources_num"]
        self._max_episode_steps = config["max_episode_steps"]
        self._sequence_length = config["sequence_length"] + 1  # add bootstrap
        self._success_distance = config["success_distance"]

        self._step_time = config["step_time"]
        self._sample_rate = config["sample_rate"]
        self._current_sample_index = 0
        self._current_source_sound, _ = librosa.load(
            self._config["audio_dir"], sr=self._sample_rate
        )

        self._stopped_agents = [False for _ in range(self._num_agents)]

        self._sim = self._get_sim()
        self._greedy_follower = [
            self._sim.make_greedy_follower(
                agent_id,
                self._success_distance,
                stop_key="stop",
                forward_key="move_forward",
                left_key="turn_left",
                right_key="turn_right",
            )
            for agent_id in range(self._num_agents)
        ]

        self._count = 0

        """predefined information"""
        self._data_structure = {
            "camera": (
                (config["resolution"], config["resolution"], 4),
                np.float32,
            ),
            "audio": (
                (2, int(self._sample_rate * self._step_time)),
                np.float32,
            ),
            "step": ((), np.int64),
            # rl_output
            "rl_pred": ((), np.int64),
            "rl_logits": ((4,), np.float32),
            "rl_value": ((1,), np.float32),
            # reward & mask
            "reward": ((), np.float32),
            "mask": ((), np.float32),
            # lstm hidden state
            "lstm_h": ((self._config["hid_dim_l"],), np.float32),
            "lstm_c": ((self._config["hid_dim_l"],), np.float32),
        }

        self._data = [
            {k: list() for k in self._data_structure.keys()}
            for _ in range(self._num_agents)
        ]

        self._path_count = [0 for _ in range(self._num_agents)]

    @staticmethod
    def action_id_2_str(action_id):
        return MultiAudioEnv.id2str[action_id]

    @staticmethod
    def action_str_2_id(action_str):
        return MultiAudioEnv.str2id[action_str]

    def _get_sim(self):
        """
        get simulator with multi-agent added, each with an audio sensor and a camera sensor

        Return:
            sim: habitat_sim.Simulator
        """
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self._config["scene_dir"]
        backend_cfg.scene_dataset_config_file = self._config["scene_config_file"]
        backend_cfg.load_semantic_mesh = self._config["load_semantic_mesh"]
        backend_cfg.enable_physics = self._config["enable_physics"]
        backend_cfg.random_seed = self._config["random_seed"]

        agent_cfg_list = []
        for _ in range(self._num_agents):
            agent_cfg = habitat_sim.agent.AgentConfiguration()

            agent_cfg.action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward",
                    habitat_sim.agent.ActuationSpec(
                        amount=self._config["forward_amount"]
                    ),
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
                "stop": habitat_sim.agent.ActionSpec("stop", StopSpec()),
            }

            # add camera sensor
            camera_sensor_spec = habitat_sim.CameraSensorSpec()
            camera_sensor_spec.uuid = "color_sensor"
            camera_sensor_spec.resolution = [self._config["resolution"]] * 2
            camera_sensor_spec.postition = [0.0, 0.0, 0.0]
            camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            agent_cfg.sensor_specifications = [camera_sensor_spec]

            agent_cfg_list.append(agent_cfg)

        cfg = habitat_sim.Configuration(backend_cfg, agent_cfg_list)

        sim = habitat_sim.Simulator(cfg)

        for agent_id in range(self._num_agents):
            # add audio sensor
            for audio_sensor_id in range(self._num_sources):
                audio_sensor_spec = habitat_sim.AudioSensorSpec()
                audio_sensor_spec.uuid = "audio_sensor_{}".format(audio_sensor_id)
                audio_sensor_spec.enableMaterials = True
                audio_sensor_spec.channelLayout.type = (
                    habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
                )
                audio_sensor_spec.channelLayout.channelCount = 1
                # audio sensor location set with respect to the agent
                audio_sensor_spec.position = [0.0, 0.0, 0.0]  # height of 1.5m
                audio_sensor_spec.acousticsConfig.sampleRate = 48000
                # whether indrect (reverberation) is present in the rendered IR
                audio_sensor_spec.acousticsConfig.indirect = True

                sim.add_sensor(audio_sensor_spec, agent_id=agent_id)
                audio_sensor = sim.get_agent(agent_id)._sensors[
                    "audio_sensor_{}".format(audio_sensor_id)
                ]
                audio_sensor.setAudioMaterialsJSON(self._config["AudioMaterialsJSON"])

        return sim

    def _reset_audio(self):
        """
        Do the following:
            1. generate random navigable source position
            2. set audio source transform for each agent, in the order of audio sensor uuid
            3. reset the audio sample index to the beginning

        """

        self._source_poses = [
            self._sim.pathfinder.get_random_navigable_point()
            for _ in range(self._num_sources)
        ]
        print(f"source_poses {self._source_poses}")
        for agent_id in range(self._num_agents):
            for audio_sensor_id in range(self._num_sources):
                audio_sensor = self._sim.get_agent(agent_id)._sensors[
                    "audio_sensor_{}".format(audio_sensor_id)
                ]
                audio_sensor.setAudioSourceTransform(
                    self._source_poses[audio_sensor_id] + np.array([0.0, 0.0, 0.0])
                )  # add height of 1.5m

        self._current_sample_index = 0
        # [chunked_audios.reset() for chunked_audios in self._chunked_audios]

    def _reset_agent(self):
        """
        randomly reset every agent position and rotation, at least 1m away from any source
        """
        # create a queue for each agent
        self._stopped_agents = [False for _ in range(self._num_agents)]

        for agent_id in range(self._num_agents):
            print(f"agent {agent_id} reseting")
            agent = self._sim.get_agent(agent_id)
            agent_state = habitat_sim.AgentState()
            while True:
                rand_pos = self._sim.pathfinder.get_random_navigable_point()
                if (
                    (
                        np.linalg.norm(rand_pos - self._source_poses[0])
                        > self._success_distance + 0.2
                    )
                    and (
                        np.linalg.norm(rand_pos - self._source_poses[0])
                        < self._success_distance + 6.0
                    )
                    and (
                        self.shortest_path(rand_pos, self._source_poses[0]) is not None
                    )
                    and (self._sim.pathfinder.is_navigable(rand_pos))
                ):
                    agent_state.position = rand_pos
                    break
                print(f"agent {agent_id} initialization retried.")
            print(f"agent_state.position {agent_state.position}")

            # Generate random yaw angle from -180 to 180
            # rand_quat = random_quat()
            # agent_state.rotation = rand_quat
            # print(f"agent_state.rotation {agent_state.rotation}")

            agent.set_state(agent_state)

        self._crushed_agents = [False] * self._num_agents

        self._path_count = [0 for _ in range(self._num_agents)]

    def _convolve_with_ir(self, ir):
        ir = ir.T
        sampling_rate = self._sample_rate
        num_sample = int(sampling_rate * self._step_time)
        index = self._current_sample_index
        if index - ir.shape[0] < 0:
            sound_segment = self._current_source_sound[: index + num_sample]
            binaural_convolved = np.array(
                [
                    fftconvolve(sound_segment, ir[:, channel])
                    for channel in range(ir.shape[-1])
                ]
            )
            audiogoal = binaural_convolved[:, index : index + num_sample]
        else:
            # include reverb from previous time step
            if index + num_sample < self._current_source_sound.shape[0]:
                sound_segment = self._current_source_sound[
                    index - ir.shape[0] + 1 : index + num_sample
                ]
            else:
                wraparound_sample = (
                    index + num_sample - self._current_source_sound.shape[0]
                )
                sound_segment = np.concatenate(
                    [
                        self._current_source_sound[index - ir.shape[0] + 1 :],
                        self._current_source_sound[:wraparound_sample],
                    ]
                )

            # sound_segment = self._current_source_sound[index - ir.shape[0] + 1: index + num_sample]
            binaural_convolved = np.array(
                [
                    fftconvolve(
                        sound_segment,
                        ir[:, channel],
                        mode="valid",
                    )
                    for channel in range(ir.shape[-1])
                ]
            )
            audiogoal = binaural_convolved

        # audiogoal = np.array([fftconvolve(self._current_source_sound, ir[:, channel], mode='full',
        #                                   ) for channel in range(ir.shape[-1])])
        # audiogoal = audiogoal[:, self._episode_step_count * num_sample: (self._episode_step_count + 1) * num_sample]
        # audiogoal = np.pad(audiogoal, [(0, 0), (0, sampling_rate - audiogoal.shape[1])])
        # print("audiogoal", audiogoal.shape)

        return audiogoal

    def reset(self):
        """
        reset the environment
        Return:
            s: dict, agent concatenated together
        """
        self._count = 0

        self._reset_audio()
        self._reset_agent()

        self._prev_obs = list()

        obs = self._get_observations()
        self._prev_obs = obs

        shape, dtype = self._data_structure["lstm_h"]
        # get state (including obs, mask, lstm_h, lstm_c)
        s = [
            {
                "camera": np.array(obs[i]["camera"]),
                "audio": np.array(obs[i]["audio"]),
                "step": np.array(self._count),
                "lstm_h": np.zeros(shape, dtype=dtype),
                "lstm_c": np.zeros(shape, dtype=dtype),
            }
            for i in range(self._num_agents)
        ]

        # store data for rl-train
        for i in range(self._num_agents):
            self._data[i]["lstm_h"].append(np.zeros(shape, dtype=dtype))
            self._data[i]["lstm_c"].append(np.zeros(shape, dtype=dtype))

            for k, v in s[i].items():
                shape, dtype = self._data_structure[k]
                self._data[i][k].append(np.array(v, dtype=dtype))

        self._succeeded_agents = [False] * self._num_agents

        self._prev_geo_dist = [
            self.get_geodesic_distance(agent_id) for agent_id in range(self._num_agents)
        ]

        return s

    def _get_observations(self):
        """
        get observation for each agent
        Return:
            obs: list of dict, representing observation for each agent
        """
        obs_list = []
        # t = time.time()
        all_obs = self._sim.get_sensor_observations(agent_ids=range(self._num_agents))
        # print(f"get_sensor_observations time: {time.time()-t}")
        for agent_id in range(self._num_agents):
            obs = all_obs[agent_id]
            # get audio chunk
            # in there , i use sam model to encode the image agent obtain
            # image_pre = obs["color_sensor"]
            
            # in there , i use a foundation model as AVSM
            
            irs = [
                np.array(obs["audio_sensor_{}".format(i)])
                for i in range(self._num_sources)
            ]
            # convolve audio chunks with IRs, sum over sources
            audios = [self._convolve_with_ir(ir) for ir in irs]
            
            audios = np.array(audios)  # source, channel, len
            audio = np.sum(audios, axis=0)  # channel, len
            # au = audioRun.audioRun(audio.squeeze()[0])
            # result = np.vstack((au,au))
            # append audio and camera sensor to obs
            print("sagment image!")
            image = obs["color_sensor"]
            image = image[:, :, :3]
            masks = self.mask_generator.generate(image)
            masks_sorted = sorted(masks, key=(lambda x: x['area']), reverse=True)
            img = np.ones((masks_sorted[0]['segmentation'].shape[0], masks_sorted[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in masks_sorted:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            obs_list.append(
                {
                    "camera": img,
                    "audio": audio,
                }
            )

        self._current_sample_index = (
            int(self._current_sample_index + self._sample_rate * self._step_time)
            % self._current_source_sound.shape[0]
        )

        return obs_list

    def _reward(self):
        """
        calculate reward for each agent
        Return:
            r: list of float, representing reward for each agent
        """
        # 0 if step exceeds max_episode_steps
        # 1 if agent reaches the goal

        if self._count >= self._max_episode_steps:
            r = [0] * self._num_agents
            return r

        geo_dist = [
            self.get_geodesic_distance(agent_id) for agent_id in range(self._num_agents)
        ]

        r = [
            (self._prev_geo_dist[agent_id] - geo_dist[agent_id]) * 10
            for agent_id in range(self._num_agents)
        ]

        self._prev_geo_dist = geo_dist

        # if agent reaches the goal (< success_distance), return 1
        for agent_id in range(self._num_agents):
            if (
                np.linalg.norm(
                    self._sim.get_agent(agent_id).get_state().position
                    - self._source_poses[0]
                )
                < self._success_distance
            ):
                r[agent_id] += 100

        for agent_id in range(self._num_agents):
            # if agent is crushed, return -1
            r[agent_id] -= 5 if self._crushed_agents[agent_id] else 0
            # Time
            # r[agent_id] -= 1

        return r

    def render(self):
        pass

    def _success(self):
        """
        define the success condition, check if all agents reach the goal
        """

        cond = [
            True
            if np.linalg.norm(
                self._sim.get_agent(agent_id).get_state().position
                - self._source_poses[0]
            )
            < self._success_distance
            else False
            for agent_id in range(self._num_agents)
        ]
        for i in range(len(cond)):
            if cond[i] == True:
                self._succeeded_agents[i] = True

        return all(cond)

    def step(self, a: List[dict]):
        """
        typical step function for rl
        Return:
            s: list of dict, representing state for each agent
            r: list of float, representing reward for each agent
            done: bool, whether the episode is done
            info: dict, additional info
        """

        self._count += 1

        self._crushed_agents = [False] * self._num_agents

        prev_agent_state_list = list()
        # take action
        for agent_id in range(self._num_agents):
            agent = self._sim.get_agent(agent_id)
            prev_agent_state_list.append(copy.deepcopy(agent.get_state()))

            if self._stopped_agents[agent_id]:
                continue

            action = a[agent_id]["rl_pred"]
            if action == 0:
                action = "move_forward"
            elif action == 1:
                action = "turn_left"
            elif action == 2:
                action = "turn_right"
            elif action == 3:
                action = "stop"
                self._stopped_agents[agent_id] = True

            collide = agent.act(action)
            self._crushed_agents[agent_id] = collide
            if action == 0 and not collide:
                self._path_count[agent_id] += 1.0
            # if collide:
            #     print(f"agent {agent_id} collide")

        # get observation
        obs = self._get_observations()

        for agent_id in range(self._num_agents):
            if self._crushed_agents[agent_id]:
                self._sim.get_agent(agent_id).set_state(prev_agent_state_list[agent_id])
                obs[agent_id] = self._prev_obs[agent_id]

        self._prev_obs = obs
        # get reward
        r = self._reward()
        # get done
        done = self._count >= self._max_episode_steps or self._success()
        # get info
        info = {
            "count": self._count,
            "euclidian_distance": [
                np.linalg.norm(
                    self._sim.get_agent(agent_id).get_state().position
                    - self._source_poses[0]
                )
                for agent_id in range(self._num_agents)
            ],
            "geodesic_distance": self._prev_geo_dist,
            # the angle between agent's forward direction and the direction to the goal
            "angle": [
                angle_between_quats(
                    self._sim.get_agent(agent_id).get_state().rotation,
                    quaternion.from_rotation_vector(
                        self._source_poses[0]
                        - self._sim.get_agent(agent_id).get_state().position
                    ),
                )
                for agent_id in range(self._num_agents)
            ],
            "collode": self._crushed_agents,
            "success": self._succeeded_agents,
            "actions": [a[i]["rl_pred"] for i in range(self._num_agents)],
        }

        # get state (including obs, mask, lstm_h, lstm_c)
        s = [
            {
                "camera": np.array(obs[i]["camera"]),
                "audio": np.array(obs[i]["audio"]),
                "step": np.array(self._count),
                "lstm_h": a[i]["lstm_h"],
                "lstm_c": a[i]["lstm_c"],
            }
            for i in range(self._num_agents)
        ]

        """store data for rl-train"""
        for i in range(self._num_agents):
            for k, v in a[i].items():
                if k == "lstm_h" or k == "lstm_c":
                    continue
                _, dtype = self._data_structure[k]
                self._data[i][k].append(np.array(v, dtype=dtype))

            for k, v in s[i].items():
                _, dtype = self._data_structure[k]
                self._data[i][k].append(np.array(v, dtype=dtype))

            for k, v in {
                "reward": r[i],
                "mask": float(not done),
            }.items():
                _, dtype = self._data_structure[k]
                self._data[i][k].append(np.array(v, dtype=dtype))

        return s, r, done, info

    @staticmethod
    def _pad(input_list, shape, dtype, target_length):
        if len(input_list) >= target_length:
            return np.array(
                input_list[:target_length], dtype=dtype
            )  # cut if over seq len
        pad_length = target_length - len(input_list)
        for _ in range(pad_length):
            input_list.append(np.zeros(shape, dtype=dtype))
        #     x = np.zeros(shape, dtype=dtype)
        # print("zero", x.shape)
        return np.array(input_list, dtype=dtype)

    def get_shortest_action_list(self, goal_pos=None, return_length=False):
        if goal_pos is None:
            goal_pos = self._source_poses[0]

        path = list()
        for agent_id in range(self._num_agents):
            path_ = self._greedy_follower[agent_id].find_path(goal_pos)
            path.append(path_)

        if return_length:
            return [
                path[agent_id].count(0) * self._config["forward_amount"]
                for agent_id in range(self._num_agents)
            ]

        return path

    def get_geodesic_distance(self, agent_id):
        from_pos = self._sim.get_agent(agent_id).get_state().position
        to_pos = self._source_poses[0]

        path = habitat_sim.ShortestPath()
        path.requested_start = from_pos
        path.requested_end = to_pos
        found_path = self._sim.pathfinder.find_path(path)
        path_results = (found_path, path.geodesic_distance, path.points)
        return path_results[1]

    def get_geo_distance_list(self):
        return [self.get_geodesic_distance(i) for i in range(self._num_agents)]

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

    def get_num_agents(self):
        return self._num_agents

    def get_agent_rotation(self):
        ret = list()
        for agent_id in range(self._num_agents):
            ret.append(self._sim.get_agent(agent_id).get_state().rotation)
        return ret

    def get_source_pos(self):
        return self._source_poses

    def get_agent_pos(self):
        ret = list()
        for agent_id in range(self._num_agents):
            ret.append(self._sim.get_agent(agent_id).get_state().position)
        return ret

    def get(self):
        """process self._episode into a list of {str: array((T, *))}, so the model can train rl loss"""
        res = list()

        for i in range(self._num_agents):
            out = list()
            # print(self._data)
            while len(self._data[i]["audio"]) > 1:  # 1 for bootstrap
                seq = dict()
                for k, v in self._data[i].items():
                    # print(k)
                    # print(len(v), v[0].shape)
                    shape, dtype = self._data_structure[k]
                    seq[k] = self._pad(v, shape, dtype, self._sequence_length)
                    self._data[i][k] = v[
                        self._sequence_length - 1 :
                    ]  # shift, preserve 1 since bootstrap
                out.append(seq)

        res.append(out)

        return out

    def observation_space(self, agent):
        """
        observation space including camera sensor and audio sensor.
        """
        return Dict(
            {
                "camera": Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),  # if int
                # "camera": Box(low=0, high=1, shape=(256, 256, 3), dtype=np.float32), # if float
                "audio": Box(low=-1, high=1, shape=(48000, 2), dtype=np.float32),
            }
        )

    def action_space(self, agent):
        """
        including 3 actions: forward, turn left, turn right
        """
        return Discrete(4)

    def prepare_spl(self):
        """
        Calculate SPL (Success weighted by Path Length).
        :param successes: List of binary success indicators (1 or 0).
        :return: SPL value.
        """

        shortest_distances = self.get_shortest_action_list(return_length=True)
        path_lengths = [
            self._path_count[agent_id] * self._config["forward_amount"]
            for agent_id in range(self._num_agents)
        ]
        
        return shortest_distances, path_lengths
