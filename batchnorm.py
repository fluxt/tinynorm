import torch
import einops


x = torch.randn(8, 16, 32, 32)
scale = torch.ones(1, 16, 1, 1)
bias = torch.zeros(1, 16, 1, 1)
running_mean = torch.zeros(1, 16, 1, 1)
running_var = torch.ones(1, 16, 1, 1)


def batchnorm(x, scale, bias, running_mean, running_var, momentum=0.1, eps=1e-5):
    mean = einops.reduce(x, "n c h w -> 1 c 1 1", "mean")
    var = einops.reduce(x, "n c h w -> 1 c 1 1", "var")
    y = (x - mean) / torch.sqrt(var + eps)
    y = scale * y + bias

    out_running_mean = (1 - momentum) * running_mean + momentum * mean
    out_running_var = (1 - momentum) * running_var + momentum * var

    return y, mean, var, out_running_mean, out_running_var

y, mean, var, out_running_mean, out_running_var = batchnorm(x, scale, bias, running_mean, running_var)


def batchnorm_inference(x, scale, bias, running_mean, running_var, eps=1e-5):
    y = (x - running_mean) / torch.sqrt(running_var + eps)
    y = scale * y + bias
    return y

y_inference = batchnorm_inference(x, scale, bias, running_mean, running_var)


def batchnorm_backward():
    pass
