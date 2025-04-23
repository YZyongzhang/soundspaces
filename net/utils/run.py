import sys ,os
sys.path.append('/home/getuanhui/project/sound-spaces/')
from yz.net.utils import trans_to_database_from_rawdata
from yz.net.utils import to_stateencode_database
from yz.config import agent_config


def get_lmdb_database():
    # muti_env_data
    T = trans_to_database_from_rawdata(store_path="env")
    # from_path = f'{agent_config.EXIT_ENV_DATA}' + 'muti_env_data'
    from_path = '/home/getuanhui/project/sound-spaces/yz/env_data/train_split'
    # muti_env_data_advance_stop
    # T = trans_to_database_from_rawdata(store_path="muti_env_advance_stop")
    # from_path = f'{agent_config.EXIT_ENV_DATA}' + 'muti_env_advance_stop'
    # muti_env_crused
    # T = trans_to_database_from_rawdata(store_path="muti_env_crushed")
    # from_path = f'{agent_config.EXIT_ENV_DATA}' + 'muti_env_crushed'
    paths = [os.path.join(from_path , i) for i in os.listdir(from_path) if i != 'RLDATA.log']
    T.get_trans(paths)
    
def get_encode_database():
    from_path = f'{agent_config.RELATIVE_DATABASE_DIR}'+'env'
    data_paths = [os.path.join(from_path , i) for i in os.listdir(from_path)]
    data_paths = sorted(data_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
    T = to_stateencode_database()
    for levelnum , data_path in enumerate(data_paths):
        T.gen_database(database_path=data_path,want_gen_database='env_encode',level=f'level{levelnum}')
    #################################################################
    # from_path = f'{agent_config.RELATIVE_DATABASE_DIR}'+'muti_env_advance_stop'
    # data_paths = [os.path.join(from_path , i) for i in os.listdir(from_path)]
    # data_paths = sorted(data_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
    # T = to_stateencode_database()
    # for levelnum , data_path in enumerate(data_paths):
    #     T.gen_database(database_path=data_path,want_gen_database='muti_env_advance_stop_encode',level=f'level{levelnum}')
    ####################################################################
    # from_path = f'{agent_config.RELATIVE_DATABASE_DIR}'+'muti_env_crushed'
    # data_paths = [os.path.join(from_path , i) for i in os.listdir(from_path)]
    # data_paths = sorted(data_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
    # T = to_stateencode_database()
    # for levelnum , data_path in enumerate(data_paths):
    #     T.gen_database(database_path=data_path,want_gen_database='muti_env_crushed_encode',level=f'level{levelnum}')
    
    
get_lmdb_database()
get_encode_database()
    