from config import config

import ray

import random

random.seed(config["random_seed"])


class Actor:
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


def collect(logger):
    """Initialization"""
    ray.init()

    num_actors = config["num_actors"]
    num_gpus = 1 / num_actors
    actors = [
        ray.remote(num_cpus=6, num_gpus=num_gpus)(Actor).remote(config)
        for _ in range(num_actors)
    ]

    seq_list = list()  # store rl data
    """Train loop"""
    for num_episodes in range(config["num_episodes"]):
        t_start = time.time()
        logger.info(f"Episode {num_episodes}")

        token_ids = [actor.rollout.remote() for actor in actors]
        result_list = ray.get(token_ids)

        for result in result_list:
            num_envs_, seq_list_, return_, num_success = result
            seq_list += seq_list_
        path = os.path.join("data/sim_512", f"offline_episode_512_{num_episodes + 181}.pkl")
        with open(path, "wb") as f:
            pickle.dump(seq_list, f)
        seq_list.clear()

        # seq num
        logger.info(f"Episode {num_episodes} seq num: {len(seq_list_)}")
        logger.info(f"Episode {num_episodes} time: {time.time()-t_start}")


if __name__ == "__main__":
    from env.v0d0 import Env
    from utils.batch import *
    import torch
    import time, pickle

    # import time, gc

    import os

    os.makedirs(config["base_dir"], exist_ok=True)
    os.makedirs(config["base_dir"] + "ckpt/", exist_ok=True)

    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config["base_dir"] + "episode.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.INFO)

    collect(logger)
