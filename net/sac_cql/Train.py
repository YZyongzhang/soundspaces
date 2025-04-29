import torch
import numpy as np
from torch.utils.data import DataLoader 
import os , math
import sys
sys.path.append('/home/getuanhui/project/sound-spaces')
from yz.config import agent_config , config

from yz.net import use_combinencode_level_data_advance_stop as Data
from yz.net import use_combinencode_data as Val_Data
from yz.net.utils import lmdb_sampler_time_seq as lmdb_sampler
from yz.net.sac_cql.SAC_CQL import CQLSAC as SAC_CQL
from yz.net.sac_cql.trainer.muti_env_train import Train as train


def begin(ckpt_dir,writer):
    database_dir = agent_config.RELATIVE_DATABASE_DIR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql = SAC_CQL(state_size=64,action_size=4,gru_inputsize=128,gru_hidden_size=64,device=device)
    database_path = [
                    '/home/getuanhui/project/sound-spaces/yz/soundspaces_data/database/noise_train_split_encode',
                    ]
    
    init_sample_dict = {
        'level0':1,
        'level1':1,
        'level2':1,
    }
    sampler = lmdb_sampler(database_path ,  shuffle=True)
    sampler.sample_data(sample=init_sample_dict)
    dataset = Data()
    train_dataloader = DataLoader(dataset=dataset,\
        sampler=sampler ,batch_size=1)
    
    val_data_base = f'{database_dir}/val_database_combinencode/'
    val_dataloader = DataLoader(dataset=Val_Data(val_data_base) , batch_size=256)
    
    num_epochs = 1000
    episode = 0
    for epoch in range(num_epochs):
        sampler_data(epoch , sampler)
        # if epoch > 50 and epoch % 10 == 0 and epoch < 150:
        #     if 0.2 * ( (epoch - 50) / 10 ) <= 1:
        #         level_1_rate = 0.2 * ( (epoch - 50) / 10 )
        #     else:
        #         level_1_rate = 1
        #     sample_dict = {
        #         'level0':1,
        #         'level1':level_1_rate,
        #         'level2':0,
        #     }
        #     sampler.sample_data(sample=sample_dict)
        #     dataloader = DataLoader(dataset=dataset,\
        #             sampler=sampler ,batch_size=1)
        
        # if epoch > 150 and epoch % 10 == 0 and epoch < 250:
        #     if 0.2 * ( (epoch - 200) / 10 ) <= 1:
        #         level_2_rate = 0.2 * ( (epoch - 200) / 10 )
        #     else:
        #         level_2_rate = 1
        #     sample_dict = {
        #         'level0':1,
        #         'level1':1,
        #         'level2':level_2_rate,
        #     }
        #     sampler.sample_data(sample=sample_dict)
        #     dataloader = DataLoader(dataset=dataset,\
        #             sampler=sampler ,batch_size=1)
        train_loss_dict = train(model=cql,dataloader=train_dataloader,)
        print(f'epoch:{epoch} , episode {episode}' + ",".join([f"{k}: {v}" for k, v in loss_dict.items()]))
        for name, item in loss_dict.items():
            writer.add_scalar(f'loss/{name}', item, episode)
        if epoch % 100 == 0 and epoch != 0:
            torch.save(cql.state_dict(), f'{ckpt_dir}/shuffle_muti_env_cql_dn_combinencode_level_0_and_1_2_{episode}_{epoch}.pth')
    torch.save(cql.state_dict(), f'{ckpt_dir}/shuffle_mutienv_cql_dn_combinencode_level_0_and_1_2.pth')

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
        f.write(f'\n{time_stamp} , message: debug the code ')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    writer = SummaryWriter(loss_dir)
    begin(ckpt_dir,writer)