import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./experiments/debug/")
    # parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=43)
    # parser.add_argument("--save_model_every", type=int, default=2)

    # ray args
    parser.add_argument("--num_actors", type=int, default=4)
    parser.add_argument("--num_envs_per_actor", type=int, default=2)

    # sample args
    # parser.add_argument("--best_eps", type=float, default=0.1)
    parser.add_argument("--save_data_every", type=int, default=10)

    # sim args
    parser.add_argument("--agents_num", type=int, default=2)
    parser.add_argument("--sources_num", type=int, default=1)
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="../data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
    )
    parser.add_argument(
        "--scene_config_file",
        type=str,
        default="../data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
    )
    parser.add_argument("--load_semantic_mesh", type=bool, default=True)
    parser.add_argument("--enable_physics", type=bool, default=False)
    parser.add_argument(
        "--AudioMaterialsJSON", type=str, default="../data/scene_datasets/mp3d_material_config.json"
    )

    # sensor args
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--sample_rate", type=float, default=24000)

    # env args
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--step_time", type=float, default=0.75)
    parser.add_argument("--success_distance", type=float, default=1.0)
    parser.add_argument("--audio_dir", type=str, default="../res/singing.wav")
    parser.add_argument("--forward_amount", type=float, default=0.25)

    # architecture args
    parser.add_argument("--hid_dim_l", type=int, default=256)# 512
    parser.add_argument("--hid_dim_p", type=int, default=128)
    parser.add_argument("--hid_dim_v", type=int, default=128)

    # train args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--pi_coef", type=float, default=1.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--vf_clip", type=float, default=0.0)
    parser.add_argument("--ppo_eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--advantage_normalization", type=bool, default=False)
    parser.add_argument("--ent_coef", type=float, default=0.00)

    args = parser.parse_args()

    return vars(args)


config = parse_args()

# config = {
#     "base_dir": "experiments/debug/",
#     "num_envs": 8,
#     "model_dir": "",
#     "num_episodes": 10000,
#     "random_seed": 42,
#     "save_model_every": 2,
#     "num_actors": 4,
#     "num_envs_per_actor": 2,
#     "best_eps": 0.1,
#     "save_data_every": 10,
#     "agents_num": 2,
#     "sources_num": 1,
#     "scene_dir": "../data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
#     "scene_config_file": "../data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
#     "load_semantic_mesh": True,
#     "enable_physics": False,
#     "AudioMaterialsJSON": "../data/mp3d_material_config.json",
#     "resolution": 256,
#     "sample_rate": 24000,
#     "max_episode_steps": 100,
#     "sequence_length": 60,
#     "step_time": 0.75,
#     "success_distance": 0.5,
#     "audio_dir": "../res/singing.wav",
#     "forward_amount": 0.25,
#     "hid_dim_l": 512,
#     "hid_dim_p": 512,
#     "hid_dim_v": 512,
#     "batch_size": 16,
#     "adam_beta2": 0.999,
#     "lr": 1e-3,
#     "warmup": 1000,
#     "max_grad_norm": 40.0,
#     "pi_coef": 1.0,
#     "vf_coef": 0.5,
#     "vf_clip": 0.0,
#     "ppo_eps": 0.2,
#     "gamma": 0.99,
#     "gae_lambda": 0.95,
#     "advantage_normalization": False,
#     "ent_coef": 0.00,
# }
