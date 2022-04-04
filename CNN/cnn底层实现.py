import torch
from torch import nn
import d2l


def corr2d(X, K):
    """
    计算二维互相关运算
    :param X: 要计算的图像
    :param K: 卷积核
    :return:
    """
    height, width = K.shape
    new_size = (X.shape[0] - height + 1, X.shape[1] - width + 1)
    Y = torch.zeros(new_size)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            Y[i, j] = (X[i:i + height, j:j + width] * K).sum()
    return Y


X = torch.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0]
])
kernel = torch.tensor([
    [0.0, 1.0],
    [2.0, 3.0]
])

corr2d(X, kernel)