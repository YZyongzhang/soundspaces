from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box, Dict
import habitat_sim.sim
import soundspaces as ss
import numpy as np
import quaternion
import copy

from utils.angles import angle_diff, quat_to_angle
from utils.audio import ChunkedAudio

from scipy.signal import fftconvolve


class MultiAudioEnv(ParallelEnv):
    metadata = {"name": "multi_audio_nav"}

    def __init__(self, config: dict):
        # deep copy config
        self._config = config.copy()
        self._num_agents = config["agents_num"]
        self._num_sources = config["sources_num"]
        self._max_episode_steps = config["max_episode_steps"]
        self._sequence_length = config["sequence_length"] + 1  # add bootstrap
        self._scuccess_distance = config["success_distance"]

        self._step_time = config["step_time"]
        self._chunked_audios = [
            ChunkedAudio(
                config["audio_dir"], config["sample_rate"], self._step_time
            )  # TODO config for chunked audio to enable multiple sources
        ] * self._num_sources

        self._max_ir_len = 70000

        self._sim = self._get_sim()

        self._count = 0

        """predefined information"""
        self._data_structure = {
            "camera": (
                (self._num_agents, 3, config["resolution"], config["resolution"]),
                np.float32,
            ),
            "audio": (
                (self._num_agents, 2, self._config["sample_rate"] * self._step_time),
                np.float32,
            ),
            "step": ((1,), np.int64),
            # rl_output
            "rl_pred": ((self._num_agents, 1), np.int64),
            "rl_logits": ((self._num_agents, 3), np.float32),
            "rl_value": ((self._num_agents, 1), np.float32),
            # reward & mask
            "reward": ((self._num_agents, 1), np.float32),
            "mask": ((self._num_agents, 3), np.float32),
            # lstm hidden state
            "lstm_h": ((self._num_agents, self._config["hid_dim_l"]), np.float32),
            "lstm_c": ((self._num_agents, self._config["hid_dim_l"]), np.float32),
        }

        self._data = {k: list() for k in self._data_structure.keys()}

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

        agent_cfg_list = []
        for _ in range(self._num_agents):
            agent_cfg = habitat_sim.agent.AgentConfiguration()

            agent_cfg.action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
            }

            # add camera sensor
            camera_sensor_spec = habitat_sim.CameraSensorSpec()
            camera_sensor_spec.uuid = "color_sensor"
            camera_sensor_spec.resolution = [self._config["resolution"]] * 2
            camera_sensor_spec.postition = [0.0, 1.5, 0.0]
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
                audio_sensor_spec.position = [0.0, 1.5, 0.0]  # height of 1.5m
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
            3. reset the audio chunk to the beginning

        """
        self._source_poses = [
            self._sim.pathfinder.get_random_navigable_point()
            for _ in range(self._num_sources)
        ]
        for agent_id in range(self._num_agents):
            for audio_sensor_id in range(self._num_sources):
                audio_sensor = self._sim.get_agent(agent_id)._sensors[
                    "audio_sensor_{}".format(audio_sensor_id)
                ]
                audio_sensor.setAudioSourceTransform(
                    self._source_poses[audio_sensor_id] + np.array([0.0, 1.5, 0.0])
                )  # add height of 1.5m

        [chunked_audios.reset() for chunked_audios in self._chunked_audios]

    def _reset_agent(self):
        """
        randomly reset every agent position and rotation, at least 1m away from any source
        """
        for agent_id in range(self._num_agents):
            agent = self._sim.get_agent(agent_id)
            agent_state = habitat_sim.AgentState()
            # TODO not valid random
            agent_state.position = self._sim.pathfinder.get_random_navigable_point()
            # TODO random face direction needed
            # rand_q = np.random.normal(size=4)
            # rand_q /= np.linalg.norm(rand_q)
            # agent_state.rotation = quaternion.as_quat_array(np.array(rand_q))
            agent.set_state(agent_state)

        self._crushed_agents = [False] * self._num_agents

    def _convolve_with_rir(self, rir):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        num_sample = int(sampling_rate * self.config.STEP_TIME)

        index = self._current_sample_index
        if index - rir.shape[0] < 0:
            sound_segment = self.current_source_sound[: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel]
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved[:, index: index + num_sample]
        else:
            # include reverb from previous time step
            if index + num_sample < self.current_source_sound.shape[0]:
                sound_segment = self.current_source_sound[index - rir.shape[0] + 1: index + num_sample]
            else:
                wraparound_sample = index + num_sample - self.current_source_sound.shape[0]
                sound_segment = np.concatenate([self.current_source_sound[index - rir.shape[0] + 1:],
                                                self.current_source_sound[: wraparound_sample]])
            # sound_segment = self.current_source_sound[index - rir.shape[0] + 1: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel], mode='valid',
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved

        # audiogoal = np.array([fftconvolve(self.current_source_sound, rir[:, channel], mode='full',
        #                                   ) for channel in range(rir.shape[-1])])
        # audiogoal = audiogoal[:, self._episode_step_count * num_sample: (self._episode_step_count + 1) * num_sample]
        audiogoal = np.pad(audiogoal, [(0, 0), (0, sampling_rate - audiogoal.shape[1])])

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

        shape, dtype = self._data_structure["lstm_h"]
        self._data["lstm_h"].append(np.zeros(shape, dtype=dtype))
        self._data["lstm_c"].append(np.zeros(shape, dtype=dtype))

        obs, mask = self._get_observations()
        self._prev_obs = obs

        # get state (including obs, mask, lstm_h, lstm_c)
        s = {
            "camera": np.array([o["camera"] for o in obs]),
            "audio": np.array([o["audio"] for o in obs]),
            "mask": np.array(mask),
            "lstm_h": self._data["lstm_h"][-1],  # (num_agents, hid_dim_l)
            "lstm_c": self._data["lstm_c"][-1],  # (num_agents, hid_dim_l)
        }

        # store data for rl-train
        for k, v in s.items():
            shape, dtype = self._data_structure[k]
            self._data[k].append(np.array(v, dtype=dtype))

        return s

    def _get_observations(self):
        """
        get observation for each agent
        Return:
            obs: list of dict, representing observation for each agent
            mask: list of nd array in shape [action_space, ],
                representing whether the agent can step forward at current position
                [0, 0, 0] if can, [1, 0, 0] if cannot
        """
        obs_list = []
        mask_list = []
        all_obs = self._sim.get_sensor_observations(agent_ids=range(self._num_agents))

        for agent_id in range(self._num_agents):
            obs = all_obs[agent_id]
            # get audio chunk
            irs = [
                self._ir_reshape(np.array(obs["audio_sensor_{}".format(i)]))
                for i in range(self._num_sources)
            ]
            audio_chunks = [
                chunked_audio(self._step_time * self._count, self._count)
                for chunked_audio in self._chunked_audios
            ]
            # convolve audio chunks with IRs, sum over sources
            audios = np.array(
                [
                    np.array(
                        [
                            np.convolve(audio_chunk, ir[n_channel])
                            for audio_chunk, ir in zip(audio_chunks, irs)
                        ]
                    )
                    for n_channel in range(len(irs[0]))
                ]
            )
            audio = np.sum(audios, axis=1)

            # append audio and camera sensor to obs
            obs_list.append(
                {
                    "camera": obs["color_sensor"],
                    "audio": audio,
                }
            )

            # mask
            agent = self._sim.get_agent(agent_id)
            agent_state = self._sim.get_agent(0).get_state()
            forward_pos = agent_state.position + np.array([2, 0, 0])
            navigable = self._sim.pathfinder.is_navigable(forward_pos)
            self._crushed_agents[agent_id] = not navigable
            mask_list.append(np.array([0, 0, 0]) if navigable else np.array([1, 0, 0]))

        return obs_list, mask_list

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

        # if agent reaches the goal (< success_distance), return 1
        r = [
            100
            if np.linalg.norm(
                self._sim.get_agent(agent_id).get_state().position
                - self._source_poses[0]
            )
            < self._scuccess_distance
            else 0
            for agent_id in range(self._num_agents)
        ]

        for agent_id in range(self._num_agents):
            # if agent is crushed, return -1
            r[agent_id] -= 10 if self._crushed_agents[agent_id] else 0
            # Time
            r[agent_id] -= 1

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
            < self._scuccess_distance
            else False
            for agent_id in range(self._num_agents)
        ]

        return any(cond)

    def step(self, a):
        """
        typical step function for rl
        Return:
            obs: list of dict, representing observation for each agent
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

            action = a["rl_pred"][agent_id]
            if action == 0:
                action = "move_forward"
            elif action == 1:
                action = "turn_left"
            elif action == 2:
                action = "turn_right"

            agent.act(action)

        # get observation
        obs, mask = self._get_observations()

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
            "distance": [
                np.linalg.norm(
                    self._sim.get_agent(agent_id).get_state().position
                    - self._source_poses[0]
                )
                for agent_id in range(self._num_agents)
            ],
            # the angle between agent's forward direction and the direction to the goal
            "angle": [
                angle_diff(
                    self._sim.get_agent(agent_id).get_state().rotation,
                    quaternion.from_rotation_vector(
                        self._source_poses[0]
                        - self._sim.get_agent(agent_id).get_state().position
                    ),
                )
                for agent_id in range(self._num_agents)
            ],
        }

        # get state (including obs, mask, lstm_h, lstm_c)
        s = {
            "camera": np.array([o["camera"] for o in obs]),
            "audio": np.array([o["audio"] for o in obs]),
            "mask": np.array(mask),
            "lstm_h": a["lstm_h"],  # (num_agents, hid_dim_l)
            "lstm_c": a["lstm_c"],  # (num_agents, hid_dim_l)
        }

        """store data for rl-train"""
        # concatenate every agent's data into one batch
        for k, v in a.items():
            if k == "lstm_h" or k == "lstm_c":
                continue
            shape, dtype = self._data_structure[k]
            self._data[k].append(np.array(v, dtype=dtype))

        for k, v in s.items():
            shape, dtype = self._data_structure[k]
            self._data[k].append(np.array(v, dtype=dtype))

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
        return np.array(input_list, dtype=dtype)

    def _ir_reshape(self, ir):
        # shape ir to max_ir_len, no matter longer or shorter
        if len(ir) > self._max_ir_len:
            ir = ir[: self._max_ir_len]
        else:
            ir = np.pad(ir, (0, self._max_ir_len - len(ir)), "constant")

    def get(self):
        """process self._episode into a list of {str: array((T, *))}, so the model can train rl loss"""
        out = list()

        while len(self._data["img_feat"]) > 1:  # 1 for bootstrap
            seq = dict()
            for k, v in self._data.items():
                shape, dtype = self._data_structure[k]
                seq[k] = self._pad(v, shape, dtype, self._sequence_length)
                self._data[k] = v[
                    self._sequence_length - 1 :
                ]  # shift, preserve 1 since bootstrap
            out.append(seq)
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
        return Discrete(3)
