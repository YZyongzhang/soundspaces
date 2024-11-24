from config import config
import random
from torch.utils.data import DataLoadr
from data.Data import File , Data
random.seed(config["random_seed"])


class Actor:
    def __init__(self, config):
        self._config = config
        self._num_episodes = 0
        self.model = Model(config)

def train(logger, writer):
    actor = Actor(config)
    directory_path = './data/sim_512/'
    files = File(directory_path).store_file()[15:30]
    for epoch in range(10):
        batch = DataLoadr(files , batch_size = 1)
        loss_dict = actor.model.learn(batch)
        for k, v in loss_dict.items():
            writer.add_scalar("rl/" + k, v)
            logger.info(f"------- {k}: {v}")
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

    writer = SummaryWriter(log_dir=(config["base_dir"] + "log/log8"))

    train(logger, writer)

