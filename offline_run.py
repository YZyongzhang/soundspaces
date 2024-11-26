from config import config
import random
import pickle
from torch.utils.data import DataLoader
from data.Data import File , Data
random.seed(config["random_seed"])


class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0
        self.model = Model(config)

def train(logger, writer):
    i = 0
    actor = Actor(config)
    directory_path = './data/sim_512/'
    files = File(directory_path).store_file()[15:]
    for file in files:
        with open(file , 'rb') as f:
            data = pickle.load(f)
        loss_dic = actor.model.learn(data)
        for k,v in loss_dic.items():
            writer.add_scalar("rl/"+k,v,i)
        i+=1
    # files = Data(files)
    # for epoch in range(1):
    #     batch = DataLoader(files , batch_size = 1)
    #     for data in batch:
    #         loss_list = actor.model.learn(data)
    #         for k,v in loss_list.items():
    #             writer.add_scalar("rl/" + k, v, i)
    #             logger.info(f"------- {k}: {v}")
    #             i+=1
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

    writer = SummaryWriter(log_dir=("log/seg1"))

    train(logger, writer)
