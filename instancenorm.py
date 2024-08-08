import torch
import einops

x = torch.randn(8, 16, 32, 32)
scale = torch.ones(1, 16, 1, 1)
bias = torch.zeros(1, 16, 1, 1)


def instancenorm(x, scale, bias, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> n c 1 1", torch.mean)
    var = einops.reduce(x, "n c h w -> n c 1 1", torch.var)
    y_hat = (x - mean) / torch.sqrt(var + eps)
    y = scale * y_hat + bias
    return y, mean, var

y, mean, var = instancenorm(x, scale, bias)


def instancenorm_inference(x, scale, bias, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> n c 1 1", torch.mean)
    var = einops.reduce(x, "n c h w -> n c 1 1", torch.var)
    y_hat = (x - mean) / torch.sqrt(var + eps)
    y = scale * y_hat + bias
    return y

y_inference = instancenorm_inference(x, scale, bias)


def instancenorm_backward():
    pass
