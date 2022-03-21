import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch

import d2l


class Accumulator:
    '''在n个变量上累加'''

    def __init__(self, n):
        self.data = [.0] * n

    def add(self, *args):
        num = 0
        for arg in args:
            self.data[num] += float(arg)
            num += 1
        # 添加数据
        # self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 归零
        self.data = [.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)


class Net:

    def __init__(self):
        self.count_times = 0

    @abstractmethod
    def forward(self, X):
        self.count_times += 1
        pass

    def get_train_times(self):
        return self.count_times

    @property
    @abstractmethod
    def params_names(self):
        pass


class Optimizer:
    """Updator = Optimizer 训练器,指定使用什么损失函数"""

    def __init__(self, net, batch_size, lr):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr

    @abstractmethod
    def step(self):
        pass


def save_params(net, path='/netParams'):
    path = os.getcwd() + path
    print(path)
    for names, values in net.params_names:
        if not os.path.exists(path):
            os.makedirs(path)
        pd.DataFrame(values.detach().numpy()).to_csv('netParams/' + names + '.CSV', index=False)  # 不保存列名
    print("lmy.py--save_params()--写入成功")


def get_params(net, path='/netParams'):
    path = os.getcwd() + path
    for name, _ in net.params_names:
        if 'b' in name:
            setattr(net, name,
                    torch.from_numpy(np.array(pd.read_csv(path + '/' + name + '.CSV'))).type(torch.float).flatten())
        else:
            setattr(net, name, torch.from_numpy(np.array(pd.read_csv(path + '/' + name + '.CSV'))).type(torch.float))
    return net
