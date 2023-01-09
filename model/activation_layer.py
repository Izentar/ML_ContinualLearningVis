import torch
from torch import nn
import math

def GaussAL(x):
    return torch.exp(-x*x)


def Conjuction(x: torch.TensorType, dim=-1):
    dimSize = x.shape[dim]
    isEven = (dimSize % 2 == 1)
    rangeEven = torch.arange(0, dimSize - isEven, step=2, device=x.device, dtype=torch.int)
    rangeOdd = torch.arange(1, dimSize, step=2, device=x.device, dtype=torch.int)
    even = torch.index_select(x, dim, rangeEven)
    odd = torch.index_select(x, dim, rangeOdd)
    if(dimSize % 2 == 1):
        last = torch.index_select(x, dim, torch.tensor([dimSize - 1], device=x.device, dtype=torch.int))
        ret = torch.cat([even * odd, last], dim=dim)
        return ret
    return even * odd