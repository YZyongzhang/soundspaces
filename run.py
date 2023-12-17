from config import config


def train(logger):
    """Initialization"""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=(config["base_dir"] + "log/"))

    num_envs = config["num_envs"]

    model = Model(config)
    if config["model_dir"] != "":  # if load an existing rl model
        model.load_model(config["model_dir"])
        print("Load model from", config["model_dir"])

    envs = [Env(config, logger) for _ in range(num_envs)]

    seq_list = list()  # store rl data

    """Train loop"""
    for num_episodes in range(config["num_episodes"]):
        t_start = time.time()
        logger.info(f"Episode {num_episodes}")
        all_r_list = list()

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
        return_ = (
            sum([sum([sum(r) for r in r_list]) for r_list in all_r_list]) / num_envs
        )
        writer.add_scalar("rl/return", return_, num_episodes)
        logger.info(f"--- return: {return_}")
        
        num_success = sum([int(t) for info in info_list for t in info["success"] ])
        logger.info(f"--- succeeded agent num: {num_success}")

        # store rl data
        for env in envs:
            seq_list += env.get()
        writer.add_scalar("train/seq_num", len(seq_list), num_episodes)
        logger.info(f"------- seq num: {len(seq_list)}")

        # Train rl model
        loss_dict = model.learn(seq_list)
        for k, v in loss_dict.items():
            writer.add_scalar("rl/" + k, v, num_episodes + 1)
            logger.info(f"------- {k}: {v}")
        seq_list.clear()

        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Episode {num_episodes} time: {time.time()-t_start}")

    writer.close()


if __name__ == "__main__":
    from env.v0d0 import Env
    from policy.v0d0 import Model
    from utils.batch import *
    import torch
    import time, gc
    
    import os
    os.makedirs(config["base_dir"], exist_ok=True)
    
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=config["base_dir"]+"out.log", filemode="w", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)
    
    train(logger)
