from config import config


def train():
    """Initialization"""
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=(config["base_dir"] + "log/"))

    num_envs = config["num_envs"]

    model = Model(config)
    if config["model_dir"] != "":  # if load an existing rl model
        model.load_rl_model(config["model_dir"])
        print("Load model from", config["model_dir"])

    envs = [Env(config) for _ in range(num_envs)]

    seq_list = list()  # store rl data

    """Train loop"""
    for num_episodes in range(config["num_episodes"]):
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

            if all(done_list):
                break

        # Test and plot return
        return_ = sum([sum(r) for r in zip(*all_r_list)]) / num_envs
        writer.add_scalar("rl/return", return_, num_episodes)
        print("--- return:", return_)

        # store rl data
        for env in envs:
            seq_list += env.get()
        writer.add_scalar("train/seq_num", len(seq_list), num_episodes)

        # Train rl model
        loss_dict = model.learn_rl(seq_list)
        for k, v in loss_dict.items():
            writer.add_scalar("rl/" + k, v, num_episodes + 1)
        seq_list.clear()

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()


if __name__ == "__main__":
    from env.v0d0 import Env
    from policy.v0d0 import Model
    from utils.batch import *
    import torch
    import time, gc

    train()
