import torch
from einops import reduce

def batchnorm(x, gamma, beta, eps=1e-5):
    mean = reduce(x, 'n c h w -> c', 'mean')
    variance = reduce(x, 'n c h w -> c', 'var')
    mean = rearrange(mean, 'c -> 1 c 1 1')
    variance = rearrange(variance, 'c -> 1 c 1 1')
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    out = gamma * x_normalized + beta
    return out

x_4d = torch.randn(8, 16, 32, 32)
gamma = torch.ones(1, 16, 1, 1)
beta = torch.zeros(1, 16, 1, 1)
output_4d = batchnorm(x_4d, gamma, beta)
