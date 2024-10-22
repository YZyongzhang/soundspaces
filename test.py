import os
import re
import pickle
def collect_data_list():
    directory_path = './data'
    file_list = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
        ]
    data_temp = []
    pattern = r"offline_episode_(\d+).pkl"
    # 正则表达式匹配文件路径
    # 前十个路径
    i = 0
    j = 0
    data_256=[]
    data_512=[]
    for file in file_list[:100]:
        match = re.search(pattern, file)
        if match:
            epi = int(match.group(1))
        
        # load pickle
        with open(file, mode='rb') as f:
            data = pickle.load(f)
        print(data[0]["lstm_h"].shape)
        if(data[0]["lstm_h"].shape == (61,256)):
            path_256 = os.path.join("data/sim_256", f"offline_episode_256_{i}.pkl")
            with open(path_256,"wb") as file_:
                pickle.dump(data, file_)
            with open(file, mode='wb') as f:
                pickle.dump(data_temp,f)
            i+=1
            data_256.append(i)
        if(data[0]["lstm_h"].shape == (61,512)):
            path_512 = os.path.join("data/sim_512", f"offline_episode_512_{j}.pkl")
            with open(path_512,"wb") as file_:
                pickle.dump(data, file_)
            with open(file, mode='wb') as f:
                pickle.dump(data_temp,f)
            j+=1
            data_512.append(j)
    # 数据初始化完毕，开始进行训练
    print(len(data_256),len(data_512))
collect_data_list()