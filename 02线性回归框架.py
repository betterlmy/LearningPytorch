import random
import torch
from torch.utils import data  # 数据处理模块

from d2l import torch as d2l
import numpy as np


def synthetic_data(w, b, num_examples):
    """
    合成数据,人工制作数据
    :param w: 超参数 手动定义的线性模型参数
    :param b: 偏置
    :param num_examples:生成的样本数量
    :return:
    """
    X = torch.normal(0, 1, (num_examples, len(w)))  # X是一个均值为0,方差为1的随机数,shape =  (num_examples, len(w)
    y = torch.matmul(X, w) + b
    epsilon = torch.normal(0, 0.01, y.shape)  # 随机取得一个一个噪声
    y += epsilon
    return X, y.reshape((-1, 1))  # reshape 的-1表示为自动计算,1表示固定值


def data_iter1(batch_size, features, labels):
    """
    从多个样本中,随机筛选出一部分组成一个batch
    :param batch_size: 一个batch中包含样本的个数
    :param features:
    :param labels:
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.seed(2)
    random.shuffle(indices)

    indices = indices[:batch_size]  # 打乱顺序
    return zip(features[indices], labels[indices])


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.seed(2)
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        #
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  #

        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """
    线性模型
    :param X:
    :param w:
    :param b:
    :return:
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    计算损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    Small Gradient Descent
    小批量随机梯度下降
    :param params: 给定的参数 w 和b
    :param lr: 学习率  learnRate
    :param batch_size:
    :return:
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 除以batch_size是因为我们是对整个进行求和,需要进行归一化处理
            param.grad.zero_()


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个Torch的数据迭代器
    :param data_arrays:
    :param batch_size:
    :param is_train:如果是训练集,则需要打乱顺序
    :return:
    """
    dataSet = data.TensorDataset(*data_arrays)  # *表示将tuple中的元素进行拆解
    return data.DataLoader(dataSet, batch_size, shuffle=is_train)


if __name__ == '__main__':
    batch_size = 2
    num_epochs = 3
    net = linreg
    loss = squared_loss
    lr = .03

    # 生成数据集,依旧使用手动生成
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 将生成的数据集保存为torch的TensorDataSet
    data_loader = load_array((features, labels), batch_size, is_train=False)
    while True:
        now_train = next(iter(data_loader))

