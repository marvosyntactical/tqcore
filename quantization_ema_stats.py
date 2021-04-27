"""
Exponential Moving Average calculation.
"""

import torch


def update_ema_stats(x, stats, key):
    """
    Calculates EMA/ MA for activations, based on https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    Each call calculates EMA for one layer, specified by key.
    Stats is a regular python dictionary containing EMA for multiple layers.
    :param x: activation tensor of layer
    :param stats: dictionary: EMA
    :param key: string, name/identifier of current layer
    :return: stats: updated EMA
    """

    max_val = torch.max(x).item()
    min_val= torch.min(x).item()
    # assert max_val != min_val, (max_val, (x==max_val).all())

    # if layer not yet in dictionary: create EMA for layer
    if key not in stats:
        stats[key] = {"max": max_val, "min": min_val}
    else:
        curr_max = stats[key]["max"]
        curr_min = stats[key]["min"]
        stats[key]['max'] = max(max_val, curr_max) if curr_max is not None else max_val
        stats[key]['min'] = max(min_val, curr_min) if curr_min is not None else min_val

    ema_decay = 0.9999

    if 'ema_min' in stats[key]:
        # stats[key]['ema_min'] = (1.-ema_decay) * min_val + ema_decay * stats[key]['ema_min']
        stats[key]['ema_min'] -=  (1 - ema_decay) * (stats[key]['ema_min'] - min_val)

    else:
        stats[key]['ema_min'] = min_val

    if 'ema_max' in stats[key]:
        # stats[key]['ema_max'] = (1.-ema_decay) * max_val + ema_decay * stats[key]['ema_max']
        stats[key]['ema_max'] -= (1 - ema_decay) * (stats[key]['ema_max'] - max_val)
    else:
        stats[key]['ema_max'] = max_val


    return stats

