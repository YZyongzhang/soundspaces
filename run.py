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

        self.model = Model(config)

    def reset(self, s):
        return self.model.load_model(s)

    def rollout(self, s):
        self.reset(s)
        self._num_episodes += 1

        all_r_list = list()

        envs = self.envs
        model = self.model
        config = self._config
        num_envs = self._num_envs

        input_d_list = [envs[idx].reset() for idx in range(num_envs)]
        # Generate RL training data
        while True:
            rl_output_list = model.inf(input_d_list)

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

        success_list = [int(t) for info in info_list for t in info["success"]]

        last_geo_distance = sum(
            [t for info in info_list for t in info["geodesic_distance"]]
        )

        seq_list = list()
        shortest_dis_list = list()
        path_list = list()
        for env in envs:
            seq_list += env.get()

            shortest_distances, path_lengths = env.prepare_spl()
            shortest_dis_list += shortest_distances
            path_list += path_lengths
            
        torch.cuda.empty_cache()

        for_spl = {
            "success": success_list,
            "shortest_distances": shortest_distances,
            "path_lengths": path_lengths,
        }

        return num_envs, seq_list, return_, sum(success_list), last_geo_distance, for_spl


def train(logger, writer):
    """Initialization"""
    ray.init()

    model = Model(config)
    if config["model_dir"] != "":  # if load an existing rl model
        model.load_model(config["model_dir"])
        print("Load model from", config["model_dir"])

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

        s = model.dump_model()
        token_ids = [actor.rollout.remote(s) for actor in actors]
        result_list = ray.get(token_ids)

        num_env_list = list()
        return_list = list()
        num_success_list = list()
        last_geo_distance_list = list()
        for_spl_all = {
            "success": list(),
            "shortest_distances": list(),
            "path_lengths": list(),
        }

        for result in result_list:
            num_envs_, seq_list_, return_, num_success, last_geo_distance_, for_spl_ = result
            seq_list += seq_list_
            num_env_list.append(num_envs_)
            return_list.append(return_)
            num_success_list.append(num_success)
            last_geo_distance_list.append(last_geo_distance_)
            for_spl_all["success"] += for_spl_["success"]
            for_spl_all["shortest_distances"] += for_spl_["shortest_distances"]
            for_spl_all["path_lengths"] += for_spl_["path_lengths"]

        num_envs = sum(num_env_list)
        return_ = sum(return_list) / num_envs * config["agents_num"]
        num_success = sum(num_success_list)
        last_geo_distance_avg = (
            sum(last_geo_distance_list) / num_envs * config["agents_num"]
        )
        spl = calculate_spl(
            for_spl_all["success"],
            for_spl_all["shortest_distances"],
            for_spl_all["path_lengths"],
        )
        soft_spl = calculate_soft_spl(
            for_spl_all["shortest_distances"],
            for_spl_all["path_lengths"], 
        )

        writer.add_scalar("rl/return", return_, num_episodes)
        writer.add_scalar("test/spl", spl, num_episodes)
        writer.add_scalar("test/soft_spl", soft_spl, num_episodes)
        writer.add_scalar("test/succeed_agent_count", num_success, num_episodes)
        writer.add_scalar(
            "test/last_geodesic_distance_avg", last_geo_distance_avg, num_episodes
        )
        writer.add_scalar("train/seq_num", len(seq_list), num_episodes)

        logger.info(f"--- return: {return_}")
        logger.info(f"--- spl: {spl}")
        logger.info(f"--- soft_spl: {soft_spl}")
        logger.info(f"--- succeeded agent num: {num_success}")
        logger.info(f"--- last geodesic distance avg: {last_geo_distance_avg}")
        logger.info(f"------- seq num: {len(seq_list)}")

        # Train rl model
        loss_dict = model.learn(seq_list)
        for k, v in loss_dict.items():
            writer.add_scalar("rl/" + k, v, num_episodes + 1)
            logger.info(f"------- {k}: {v}")
        seq_list.clear()

        # save model
        save_path = config["base_dir"] + f"ckpt/ckpt_{num_episodes}_ret_{return_}"
        model.save(save_path)

        logger.info(f"Episode {num_episodes} time: {time.time()-t_start}")

    writer.close()


if __name__ == "__main__":
    from env.v0d0 import Env
    from policy.v0d0 import Model
    from utils.batch import *
    from utils.metrics import *
    import torch
    import time

    # import time, gc

    import os

    os.makedirs(config["base_dir"], exist_ok=True)
    os.makedirs(config["base_dir"] + "ckpt/", exist_ok=True)

    import logging
    from torch.utils.tensorboard import SummaryWriter

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config["base_dir"] + "episode.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.INFO)

    writer = SummaryWriter(log_dir=(config["base_dir"] + "log/"))

    train(logger, writer)
