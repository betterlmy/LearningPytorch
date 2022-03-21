import random
import torch
from torch.utils import data  # 数据处理模块

import d2l1
import numpy as np
from torch import nn


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
    batch_size = 10
    num_epoches = 3
    lr = .03

    # 生成数据集,依旧使用手动生成
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 将生成的数据集保存为torch的TensorDataSet
    data_loader = load_array((features, labels), batch_size, is_train=False)

    net = nn.Sequential(nn.Linear(2, 1))  # 输入是二维的 输出是一维的
    # 使用net之前,需要初始化模型参数.例如在线性回归模型中的权重,和偏置.
    net[0].weight.data.normal_(0, .01)  # net[0]选中第一层
    net[0].bias.data.fill_(0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化算法 pytorch中optim模块定义了许多优化算吗
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 在获取完一次数据之后,进行以下步骤
    # 1 调用 net(x)生成预测并计算损失l
    # 2 通过反向传播计算梯度
    # 3 通过调用优化器更新模型参数
    for epoch in range(num_epoches):
        for X, y in data_loader:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1} loss: {l:.5f}")

    w = net[0].weight.data
    b = net[0].bias.data
    print(f"w的误差{w.reshape(true_w.shape)-true_w}")
    print(f"b的误差{b-true_b}")