import os

class data_list:
    def __init__(self):
        this.self = self
    def collect_data_list():
        directory_path = './data/'
        file_list = [
            os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
            ]
        data_d = {}
        pattern = r"offline_episode_(\d+).pkl"
        # 正则表达式匹配文件路径
        # 前十个路径
        for file in file_list[:100]:
            match = re.search(pattern, file)
            if match:
                epi = int(match.group(1))
            
            # load pickle
            with open(file, mode='rb') as f:
                data = pickle.load(f)

            data_d[epi] = data
        # 数据初始化完毕，开始进行训练
        return data_d
