from inspect import isfunction
import torch

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def percentile_loss(y_pred, y_true, percentile=99):
    """
    计算预测值和真实值之间第99百分位数的损失。
    """
    # 计算绝对误差
    errors = torch.abs(y_pred - y_true)
    
    # 计算第99百分位数的损失
    k = int(percentile / 100.0 * errors.numel())
    top_k_values, _ = torch.topk(errors.view(-1), k, sorted=True)
    loss = top_k_values[-1]  # 第99百分位数是top_k中的最后一个元素
    
    return loss

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
