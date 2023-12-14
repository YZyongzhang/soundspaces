import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="aaa/")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--num_episodes", type=int, default=1000)

    # sim args
    parser.add_argument("--agents_num", type=int, default=1)
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
        "--AudioMaterialsJSON", type=str, default="../data/mp3d_material_config.json"
    )

    # sensor args
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--sample_rate", type=float, default=48000)

    # env args
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--sequence_length", type=int, default=50)
    parser.add_argument("--step_time", type=float, default=0.25)
    parser.add_argument("--success_distance", type=float, default=0.5)
    parser.add_argument("--audio_dir", type=str, default="../res/singing.wav")

    # architecture args
    parser.add_argument("--hid_dim_l", type=int, default=512)
    parser.add_argument("--hid_dim_p", type=int, default=512)
    parser.add_argument("--hid_dim_v", type=int, default=512)

    # train args
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
