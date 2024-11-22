from config import config

import ray
import os
import re
import random
import pickle
random.seed(config["random_seed"])


class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0

        self._num_envs = config["num_envs_per_actor"]
        self.envs = [Env(config) for _ in range(self._num_envs)]

        self.model = Model(config)
def extract_number(filename):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def train(logger, writer):
    actor = Actor(config)
    i = 0
    """Train loop"""
    directory_path = './data/audio_512/'
    # directory_path_1 = './data/audio_512/'
    file_list = sorted([
    os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
    ],key=lambda f: extract_number(os.path.basename(f)))
    # file_list_1 = sorted([
    # os.path.join(directory_path_1, f) for f in os.listdir(directory_path_1) if os.path.isfile(os.path.join(directory_path_1, f))
    # ],key=lambda f: extract_number(os.path.basename(f)))
    # file_list.extend(file_list_1)
    for _ in range(10):
        for file in file_list[:len(file_list)]:
            i+=1
            with open(file, mode='rb') as f:
                print(f"{f}")
                data = pickle.load(f)
            loss_dict = actor.model.learn(data)
            for k, v in loss_dict.items():
                writer.add_scalar("rl/" + k, v, i)
                logger.info(f"------- {k}: {v}")
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

    writer = SummaryWriter(log_dir=(config["base_dir"] + "log/log2"))

    train(logger, writer)

