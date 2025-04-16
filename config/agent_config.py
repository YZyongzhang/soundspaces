import argparse

def AgentConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_PARH_COLLECT' ,type=str , default='/home/getuanhui/project/sound-spaces/yz/data/RL/')
    parser.add_argument('--MP3D_SCENE_DATASET' ,type=str , default='/home/getuanhui/project/sound-spaces/data/scene_datasets/mp3d/')
    parser.add_argument('--EXPERIMENTS_DIR' ,type=str , default='/home/getuanhui/project/sound-spaces/yz/experiments/')
    parser.add_argument('--DATABASE_DIR' ,type=str , default='/home/getuanhui/project/sound-spaces/yz/data/database/')
    

    parser.add_argument('--RELATIVE_EXPERIMENTS_DIR' ,type=str , default='./yz/experiments/')
    parser.add_argument('--RELATIVE_DATABASE_DIR' ,type=str , default='./yz/database/')
    args = parser.parse_args()

    return args


config = AgentConfig()