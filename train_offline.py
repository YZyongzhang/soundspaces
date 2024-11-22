import os
import re
import pickle
def collect_data_list():
    directory_path = './data/sim_512'
    file_list = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
        ]
    data_d = {}
    pattern = r"offline_episode_(\d+).pkl"
    # 正则表达式匹配文件路径
    # 前十个路径
    data_256=[]
    i = 0
    j = 0
    data_512=[]
    for file in file_list[:70]:
        match = re.search(pattern, file)
        if match:
            epi = int(match.group(1))
        
        # load pickle
        with open(file, mode='rb') as f:
            data = pickle.load(f)
            print(data[15]["lstm_h"].shape)
            if(data[0]["lstm_h"].shape == (61,256)):
                i+=1
                data_256.append(i)
            if(data[0]["lstm_h"].shape == (61,512)):
                j+=1
                data_512.append(j)
    # 数据初始化完毕，开始进行训练
    print(len(data_256),len(data_512))
collect_data_list()