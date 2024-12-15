from config import config
import random
import pickle
import os
import re
from torch.utils.data import DataLoader
# from data.Data import File , Data
random.seed(config["random_seed"])


class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0
        self.model = Model(config)
# def get_num(file):
#     match = re.match(r"offline_episode_audio_(\d+).pkl" , file)
#     return int(match.group(1))
def train(logger, writer):
    i = 0
    actor = Actor(config)
    directory_path = '/home/getuanhui/project/sound-spaces/yz/data/audio'
    # files = File(directory_path).store_file()[15:]
    files = os.listdir(directory_path)
    # files.sort(key= lambda x : get_num(x))
    for e in range(1):
        for file in files[:200]:
            print(file)
            path = os.path.join(directory_path , file)
            with open(path , 'rb') as f:
                data = pickle.load(f)
            loss_dic = actor.model.learn(data)
            for k,v in loss_dic.items():
                writer.add_scalar("rl/"+k,v,i)
            i+=1
if __name__ == "__main__":
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

    writer = SummaryWriter(log_dir=("log/IQL6"))

    train(logger, writer)

