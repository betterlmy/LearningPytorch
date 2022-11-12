import os
import torch


def mkdir(path):
    # 创建文件夹
    if not os.path.exists(path):
        os.mkdir(path)


def to_img(x):
    out = 0.5*(x+1)
    out = out.clamp(0, 1)  # clamp将随机变化的数值限制在给定区间内
    out = out.view(-1, 1, 28, 28)
    return out


def get_device(default = None):
    if default is not None:
        return torch.device(default)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

