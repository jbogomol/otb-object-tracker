# func_tensor.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/30/2020
#
# contains helper functions for torch.Tensor operations


import torch


def saturate_1d(t, low, high):
    """
    Saturates all values in a rank-1 tensor t to be between a specified
    minimum and maximum value.

    Args
        t: torch.Tensor, the rank-1 tensor to perform operations on
        low: int, minimum number allowed in result tensor
        high: int, maximum number allowed in result tensor

    Return the resulting torch.Tensor
    """

    t_len = len(t)
    result = torch.zeros([t_len])
    for i in range(t_len):
        val = t[i]
        if val < low:
            result[i] = low
        elif val > high:
            result[i] = high
        else:
            result[i] = val

    return result.long()
            
