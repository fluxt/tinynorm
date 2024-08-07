import torch
import einops

x = torch.randn(8, 16, 32, 32)
scale = torch.ones(1, 16, 1, 1)
bias = torch.zeros(1, 16, 1, 1)

def layernorm(x, scale, bias, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> n c 1 1", torch.mean)
    var = einops.reduce(x, "n c h w -> n c 1 1", torch.var)
    y = (x - mean) / torch.sqrt(var + eps)
    y = scale * y + bias
    return y, mean, var

y, mean, var = layernorm(x, scale, bias)


def layernorm_inference(x, scale, bias, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> n c 1 1", torch.mean)
    var = einops.reduce(x, "n c h w -> n c 1 1", torch.var)
    y = (x - mean) / torch.sqrt(var + eps)
    y = scale * y + bias
    return y

y_inference = layernorm_inference(x, scale, bias)


def instancenorm_backward():
    pass
