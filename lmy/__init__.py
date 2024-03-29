import os
import time
from abc import abstractmethod

import GPUtil
import matplotlib_inline
import numpy as np
import pandas as pd
import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from tqdm import tqdm
import sys,os
sys.path.append(os.path.abspath("../"))
# import lmy


class Timer:
    """记录多次运行时间"""

    def __init__(self, name='unnamed Timer'):
        """Defined in :numref:`subsec_linear_model`"""
        self.tik = None
        self.times = []
        self.name = name
        self.state = 'stopped'

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        print(f"{self.name} has run for {self.stop():.4f}s")

    def start(self):
        """启动计时器"""
        if self.state == 'running':
            print(f"{self.name} is still running")
        else:
            self.tik = time.time()
            self.state = 'running'

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        if self.state == 'running':
            self.times.append(time.time() - self.tik)
            self.state = 'stopped'
            return self.times[-1]
        else:
            print(self.state)
            print(f"{self.name} is not running")
            return None

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


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
    @staticmethod
    def mkdir():
        if not os.path.exists('./result'):
            os.mkdir('./result')

    def __init__(self, name='fig', xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        self.mkdir()
        if legend is None:
            legend = []
        display.set_matplotlib_formats('svg')  
        self.name = name
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
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
        plt.savefig(f'result/{self.name}.svg')


class Net:

    def __init__(self):
        self.count_times = 0

    def __call__(self, *args, **kwargs):
        for arg in args:
            self.forward(arg)

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

    @property
    @abstractmethod
    def params(self):
        pass


class Optimizer:
    """Updator = Optimizer 训练器,指定使用什么损失函数"""

    def __init__(self, net, lr, batch_size):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr

    @abstractmethod
    def step(self, net):
        pass


class SGD(Optimizer):
    """使用SGD损失函数的优化器"""

    def __init__(self, net, lr, batch_size):
        super().__init__(net, lr, batch_size)

    def step(self, net):
        self.net = net
        return sgd(self.net, self.lr, self.batch_size)


def sgd(net, lr, batch_size):
    """小批量随机梯度下降

    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for _, param in net.params_names:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def show_heatmaps(matrices,xlabel,ylabel,titiles=None,figsize=(2.5,2.5),cmap='Reds'):
    """显示矩阵热图"""
    """在后面的章节中,我们将经常调用show_heatmaps函数来显示注意力权重。"""
    d2l.use_svg_display()
    num_rows,num_cols = matrices.shape[0],matrices.shape[1]
    fig,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize,
                                sharex=True,sharey=True,squeeze=False)
    for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
        for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(),cmap=cmap)
            if i == num_rows -1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titiles:
                ax.set_title(titiles[j])
    fig.colorbar(pcm,ax=axes,shrink=.6)
    
def save_params(net, path='/netParams'):
    if isinstance(net, nn.Module):
        pass
        print("nn定义的模型类 写入失败")
        return False
    path = os.getcwd() + path
    for names, values in net.params_names:
        if not os.path.exists(path):
            os.makedirs(path)
        pd.DataFrame(values.detach().numpy()).to_csv(
            'netParams/' + names + '.CSV', index=False)  # 不保存列名
    print("写入成功", path)


def get_params(net, path='/netParams'):
    path = os.getcwd() + path
    for name, _ in net.params_names:
        if 'b' in name:
            a = torch.from_numpy(np.array(pd.read_csv(
                path + '/' + name + '.CSV'))).type(torch.float).flatten()
        else:
            a = torch.from_numpy(np.array(pd.read_csv(
                path + '/' + name + '.CSV'))).type(torch.float)
        a.requires_grad_(True)
        setattr(net, name, a)
    return net


def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, std=.1)


def cross_entropy(y_hat, y):
    """计算交叉熵"""
    # y_hat 64*10  y 64*1
    # y中存放了真实的标签下标
    # 计算交叉熵相当于对正确下标求-Log,越大越好
    #
    # test1 = y_hat[0, 1]
    # test2 = y_hat[0][1]
    x = torch.abs(y_hat[range(len(y_hat)), y])
    #
    return -torch.log(x)


def relu(X):
    """ ReLU激活函数
    :param X:
    :return:
    """
    a = torch.zeros_like(X)
    return torch.max(a, X)


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小

    Defined in :numref:`sec_calculus`"""
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()  # 配置网格线


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个数据迭代器

    Args:
        data_arrays (tuple of arrays): 包含训练数据的元组,特征和标签
        batch_size (int): 批量大小
        is_train (bool, optional): 是否是训练模式,如果是训练模式则会打乱返回的顺序. Defaults to True.

    Returns:
        DataLoader: 数据加载器
    """
    dataset = data.TensorDataset(*data_arrays)
    dl = data.DataLoader(dataset, batch_size, shuffle=is_train)
    print("数据加载完成")
    return dl


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式
    for X, y in train_iter:
        y_hat = net.forward(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater.step(net)


def print_shape(X, X_name=None):
    print(f"{X_name}.type: {type(X)}")
    if hasattr(X, 'shape'):
        print(f"{X_name}.shape = {X.shape}")
    else:
        print(f"{X_name} has no attribute of shape")
    print("*" * 20)


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


def loadFashionMnistData(batch_size, root="../lmy/data", resize=None, trans=None, num_workers=0):
    """下载FashionMnist数据集并加载到内存中
    :param num_workers:
    :param root:
    :param batch_size:
    :param resize:
    :param trans:
    :return:返回训练集和测试集的DataLoader
    """
    # 通过ToTenser()这个类 将图像数据从PIL类型转为浮点型的tensor类型,并除以255使得所有的像素数值均在0-1之间(归一化)
    transform = None
    if not trans:
        trans = [transforms.ToTensor()]
        transform = transforms.Compose(trans)
    if resize:
        trans.insert(0, transforms.Resize(resize))
        transform = transforms.Compose(trans)
    print(os.path.abspath(root))

    mnist_train = FashionMNIST(
        root=root, train=True, transform=transform, download=False)

    mnist_test = FashionMNIST(root=root, train=False,
                              transform=transform, download=False)
    print(
        f"FashionMNIST数据集加载成功，训练集大小:{len(mnist_train)},测试集大小:{len(mnist_test)},数据集shape:{mnist_test[0][0].shape}")  # 60000 ,10000
    # print_shape(mnist_test)
    return (data.DataLoader(mnist_train, batch_size, shuffle=False, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def evaluate_accuracy_gpu(net, data_iter, device=None, timer=None):
    time_available = False
    if timer is not None:
        time_available = True
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)
    if time_available and timer.state == 'stopped':
        timer.start()
    with torch.no_grad():
        for X, y in data_iter:
            if time_available:
                timer.stop()
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            timer.start()
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def getGPU(utilRateLimit=.3, contain_cpu=False):
    """
    获取所有的gpu（包括CPU）
    :param utilRateLimit: 使用率限制
    :param contain_cpu: 是否包含CPU，如果包含则返回值第一个为CPU
    :return: devices和names
    """
    devices = []
    if contain_cpu:
        devices.append(torch.device('cpu'))
    # if torch.backends.mps.is_available():
    #     devices.append(torch.device('mps'))
    #     return devices
    if not torch.cuda.is_available():
        return devices
    for gpu in GPUtil.getGPUs():
        if gpu.memoryUtil < utilRateLimit:
            """仅挑选GPU使用率小于30%"""
            devices.append(torch.device(f'cuda:{gpu.id}'))
    return devices


def train_GPU_FASHION_MNIST(net, num_epochs, lr, batch_size=256, num_devices=1, devices=getGPU(contain_cpu=False), resize=None, net_name='net'):
    """使用GPU训练模型,数据集FASHION_MNIST

    Args:
        net (nn.Module): 要训练的网络模型
        num_epochs (int): 
        lr (int): 学习率
        batch_size (int, optional): . Defaults to 256.
        num_devices (int, optional): . Defaults to 1.
        devices (list, optional): 训练的设备列表. Defaults to getGPU(contain_cpu=False).
    """
    train_iter, test_iter = loadFashionMnistData(batch_size, resize=resize)
    train_GPU(net, train_iter, test_iter, num_epochs, lr,
              num_devices=num_devices, devices=devices, net_name=net_name)


def train_GPU(net, train_iter, test_iter, num_epochs, lr, timer=Timer(), devices=getGPU(utilRateLimit=.6, contain_cpu=False), num_devices=1,
              init_weight=init_weights, net_name="net"):
    """用GPU训练模型"""
    assert devices.__class__ == list, "devices must be a list"
    assert num_devices < 0 or num_devices.__class__ == int, "num_devices must be int or None"

    if num_devices > len(devices):
        num_devices = len(devices)
        print("设备数量不足,已自动调整")
    net.apply(init_weight)
    if num_devices == 0:
        devices = [torch.device('cpu')]
    else:
        devices = devices[:num_devices]
        net.to(devices[0])
    timer.start()
    if num_devices > 1:
        net = nn.DataParallel(net, device_ids=devices)
    print(f"训练设备{devices}")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    animator = Animator(
        name=net_name,
        xlabel='epoch',
        xlim=[1, num_epochs],
        ylim=[.0, 1.0],
        legend=['train_loss', 'train_acc', 'test_acc']
    )
    num_batches = len(train_iter)
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        if timer.state == 'stopped':
            timer.start()
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            if timer.state == 'stopped':
                timer.start()
            optimizer.zero_grad()
            timer.stop()
            if 'cuda' in devices[0].type:
                X, y = X.to(devices[0]), y.to(devices[0])
            timer.start()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0], timer)
        animator.add(epoch + 1, (None, None, test_acc))
        timer.stop()
        # print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {devices}")
    print(
        f"loss:{train_l * 100:.3f}%,train_acc:{train_acc * 100:.3f}%,test_acc:{test_acc * 100:.3f}%)")



def print_shape(*args):
    shapes = []
    for arg in args:
        if hasattr(arg,"shape"):
            shapes.append(arg.shape)
    print(f"shape:")
    for shape in shapes:
        print(shape)

def to_image(image,path=None):
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    
    pic = toPIL(image)
    if path is not None:
        pic.save(path)
    return pic
    
if __name__ == "__main__":
    pass
    # to_image(torch.tensor([[56,56],
    #                      [32,32],
    #                      [64,64]]),"./1.jpg")