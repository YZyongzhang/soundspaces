from config import config
import train_offline.py as offline_data_function
import ray

import random

random.seed(config["random_seed"])
def train(logger, writer):
    # Train rl model
    offline_data = offline_data_function.collect_data_list()
    for i in range(len(offline_data)):
        seq_list = offline_data[i]
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

