import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset , DataLoader 
from torch.utils.data._utils.collate import default_collate
import os , math
from tqdm import tqdm
import torch.optim as optim
import  pickle
import logging
import time
import pdb
import sys
import ray
sys.path.append('/home/getuanhui/project/sound-spaces')
from yz.config import agent_config , config

# from yz.net import use_combinencode_level_data as Data
from yz.net import use_combinencode_level_data_advance_stop as Data
from yz.net import use_combinencode_data as Val_Data
# from yz.net.utils import lmdb_sampler
from yz.net.utils import lmdb_sampler_advance_stop
from yz.net.sac_cql.SAC_CQL import DiscreteSAC_CQL as SAC_CQL
from yz.net.sac_cql.SAC_CQL import Critic_Actor
from yz.net.sac_cql.trainer.trainer import Train as train

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    base_dir = agent_config.RELATIVE_EXPERIMENTS_DIR
    time_stamp = "{0:%Y-%m-%d~%H-%M-%S}".format(datetime.now())
    loss_dir = base_dir +'/loss/'  + time_stamp
    log_dir = base_dir + '/log/'
    ckpt_dir = base_dir + '/ckpt/' + time_stamp
    train_message = base_dir + 'train.log'
    with open(train_message , 'a') as f:
        f.write(f'\n{time_stamp} , message: write this train message')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    writer = SummaryWriter(loss_dir)
    logging.basicConfig(filename=f'{log_dir}/{time_stamp}.log', level=logging.INFO,filemode='a')
    train(ckpt_dir,writer)