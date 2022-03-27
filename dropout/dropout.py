import torch.nn as nn
import torch
import d2l
from icecream import ic


def dropout_layer(X, dropout):
    """
    将X随机选择置0
    :param X:
    :param dropout: 丢弃概率
    :return:
    """
    assert 0 <= dropout <= 1
    if dropout == 1:
        # 丢弃概率为0 直接输出0
        return torch.zeros_like(X)
    elif dropout == 0:
        return X
    mask = torch.rand(X.shape)
    mask = (mask > dropout).float()
    return mask * X / (1 - dropout)


X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
ic(X)
ic(dropout_layer(X, .0))
ic(dropout_layer(X, .5))
ic(dropout_layer(X, 1.))
