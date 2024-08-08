import torch
import einops

x = torch.randn(8, 16, 32, 32)
scale = torch.ones(1, 16, 32, 32)
bias = torch.zeros(1, 16, 32, 32)
dy = torch.randn(8, 16, 32, 32)

def layernorm(x, scale, bias, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> n 1 1 1", torch.mean)
    var = einops.reduce(x, "n c h w -> n 1 1 1", torch.var)
    y_hat = (x - mean) / torch.sqrt(var + eps)
    y = scale * y_hat + bias
    return y, mean, var

y, mean, var = layernorm(x, scale, bias)


def layernorm_inference(x, scale, bias, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> n 1 1 1", torch.mean)
    var = einops.reduce(x, "n c h w -> n 1 1 1", torch.var)
    y_hat = (x - mean) / torch.sqrt(var + eps)
    y = scale * y_hat + bias
    return y

y_inference = layernorm_inference(x, scale, bias)


def layernorm_backward(dy, x, scale, bias, mean, var, eps=1e-5):
    y_hat = (x - mean) / torch.sqrt(var + eps)

    dscale = einops.reduce(dy * y_hat, "n c h w -> n 1 1 1", torch.sum)
    dbias = einops.reduce(dy, "n c h w -> n 1 1 1", torch.sum)
    dy_hat = dy * scale

    dvar = None # ???
    dmean = None # ???
    dx = None # ???

    return dx, dscale, dbias

dx, dscale, dbias = layernorm_backward(dy, x, scale, bias, mean, var)
