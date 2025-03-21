import os, pickle 
import numpy as np
from tqdm import tqdm
def deal_data(path):
    frist_path = os.listdir(path=path)
    files = []
    for i in frist_path:
        files.append(os.path.join(path , i))
    return files
def load_data(files):
    num_episodes = 0
    for i in files:
        path_ = os.path.join(f"../../../data/RL/deal_data",  f"rl_episode_{num_episodes}.pkl")
        with open(i,'rb') as f:
            data = pickle.load(f)
        result = trs_data(data)
        if not result:
            continue
        with open(path_, "wb") as f:
            pickle.dump(result, f)
        num_episodes+=1
def trs_data(batch_data):
    
    pre_audio = list()
    pre_visual = list()
    next_audio = list()
    next_visual = list()
    done = list()
    reward = list()
    action = list()

    data = batch_data[0]
    done_ = batch_data[2]
    frist_state = batch_data[3]
    if len(data) != 1:
        return None
    audio_ = get_data(data,'audio')
    tag_ = get_data(data,'rl_pred')
    visual_ = get_data(data,'camera')
    reward_ = get_data(data,'reward')
    next_audio = list()
    tag = list()
    next_visual = list()
    reward = list()
    
    for index,tag in enumerate(tag_):
        if tag == 3 or index == 199:
            next_audio = audio_[:index+1]
            next_visual = visual_[:index+1]
            reward = reward_[:index+1]
            tag = tag_[:index+1]
            # 不包含stop信息，如需包含请index+1。具体就是指最后执行完stop之后不再会有相同的img产生
            break
    next_audio.pop(0)
    next_visual.pop(0)
    reward.pop(0)
    tag.pop(0)
    
    current_done = [1 if d[0] else 0 for d in done_]
    
    ##  get state , next_state
    pre_audio = [frist_state[0]['audio']] + next_audio[:-1]
    pre_visual = [frist_state[0]['camera']] + next_visual[:-1]
    
    return {
            "pre_audio":pre_audio,
            "pre_visual":pre_visual, 
            "next_audio":next_audio, 
            "next_visual":next_visual,
            "done":current_done,
            "reward":reward,
            "action":tag
            }
def get_data( data , name):
    d = list()
    for i in range(len(data)):
        d.extend(data[i][name])
    return d
if __name__ == "__main__":
    path = '../../../data/RL/newdone'
    files = deal_data(path= path)
    load_data(files)