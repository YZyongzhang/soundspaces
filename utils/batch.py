import numpy as np
import random

def stack_batch(dict_list):
    """
    [..., {str: (T, *)}, ...] -> {str: (B, T, *)}
    """
    out = dict()
    keys = list(dict_list[0].keys())
    for key in keys:
        out[key] = np.stack([d[key] for d in dict_list], axis=0)
    return out


def stack_batch_multi(dict_list):
    """
    [..., {str: (T, *)}, ...] -> {str: (B, T, *)}
    """
    n_envs = len(dict_list)
    n_agents = len(dict_list[0])
    out = dict()
    keys = list(dict_list[0][0].keys())
    for key in keys:
        out[key] = np.stack(
            [dict_list[i][j][key] for i in range(n_envs) for j in range(n_agents)],
            axis=0,
        )
    return out


def unstack_batch_multi(batch_dict, n_agents):
    """
    {str: (B, T, *)} -> [..., [{str: (T, *)}, ..., {str: (T, *)}], ...]
    """
    keys = list(batch_dict.keys())
    batch_size = len(batch_dict[keys[0]])
    # batch_size = n_agents * n_envs
    n_envs = batch_size // n_agents
    assert batch_size % n_agents == 0
    out = list()
    for i in range(n_envs):
        out.append(
            [
                {k: v[i * n_agents + j] for k, v in batch_dict.items()}
                for j in range(n_agents)
            ]
        )

    return out


def env_dict_batch(l):
    out = list()
    keys = list(l[0][0].keys())
    for i in range(len(l[0])):
        out.append(dict())
        for key in keys:
            out[i][key] = list()
            for j in range(len(l)):
                out[i][key].append(l[j][i][key])

    return out


def env_batch(l):
    return [list(x) for x in zip(*l)]


def unstack_batch(batch_dict):
    """
    {str: (B, T, *)} -> [..., {str: (T, *)}, ...]
    """
    keys = list(batch_dict.keys())
    batch_size = len(batch_dict[keys[0]])
    out = list()
    for i in range(batch_size):
        out.append({k: v[i] for k, v in batch_dict.items()})
    return out


class DictLoader:
    
    def __init__(self, data, batch_size=8):
        self.data = data
        self.len = len(data)
        self.batch_size = batch_size
        self.i = 0
        random.shuffle(data)
        
    def __iter__(self):
        while self.i < self.len:
            res = self.data[self.i : self.i + self.batch_size]
            self.i += self.batch_size
            bn = self.batch_size if self.i <= self.len else self.len + self.batch_size - self.i
            yield stack_batch(res), bn
