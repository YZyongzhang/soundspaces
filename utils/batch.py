import numpy as np


def stack_batch(dict_list):
    """
    [..., {str: (T, *)}, ...] -> {str: (B, T, *)}
    """
    out = dict()
    keys = list(dict_list[0].keys())
    for key in keys:
        out[key] = np.stack([d[key] for d in dict_list], axis=0)
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
