import torch
from torch import nn
from torch.nn.functional import relu
import math
from torch import Tensor


class GaussALC(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        #self.sigma = sigma
        self.sigma = torch.nn.Parameter(torch.tensor(sigma, dtype=torch.float32, requires_grad=True))
        #self.register_buffer('sigma', sigma)

        #self.mi = torch.nn.Parameter(torch.rand(1, dtype=torch.float32, requires_grad=True))
        #self.register_buffer('mi', mi)

    def forward(self, x):
        #print(self.sigma, self.mi)
        return torch.exp(-(1/self.sigma)**2 * x*x)

def gaussA(x, sigma=0.1):
    #print(torch.sum(x))
    #print(torch.sum(-0.5 * x*x))
    r = torch.exp(-(1/sigma)**2 * x*x)
    #print(r)
    #r = relu(x)
    #print(torch.sum(r))
    #r2 = torch.exp(-0.5 * torch.square(x))
    #print(torch.sum(r))
    #print(torch.max(r), torch.min(r))
    #print()
    #exit()
    return r

class GaussA(torch.nn.Module):
    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, input: Tensor) -> Tensor:
        return gaussA(input, sigma=self.sigma)

    def extra_repr(self) -> str:
        return f'sigma={self.sigma}'


def conjunction(x: torch.TensorType, dim=-1):
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
    #print((even * odd).shape)
    #exit()
    return even * odd