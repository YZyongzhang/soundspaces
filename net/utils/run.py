import sys
sys.path.append('/home/getuanhui/project/sound-spaces/')
from yz.net.utils import trans_to_database_from_rawdata
from yz.net.utils import to_stateencode_database
from yz.config import agent_config
# print('en?')
# T = trans_to_database_from_rawdata()
# print(T)
# T.get_trans()

print("begin")
T = to_stateencode_database()
level0_database_path = agent_config.RELATIVE_DATABASE_DIR + '/muti_env_advance_stop' + '/muti_env_data_0'
level1_database_path = agent_config.RELATIVE_DATABASE_DIR + '/muti_env_advance_stop' + '/muti_env_data_1'
level2_database_path = agent_config.RELATIVE_DATABASE_DIR + '/muti_env_advance_stop' + '/muti_env_data_2'
# import pdb; pdb.set_trace()
T.gen_database(level0_database_path , level=0)
T.gen_database(level1_database_path , level=1)
T.gen_database(level2_database_path , level=2)