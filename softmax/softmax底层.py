import numpy as np
import torch
import torchvision
from memory_profiler import profile
from torch.utils import data
from torchvision import transforms
import d2l
from Timer import Timer
import IPython
from IPython import display
import pandas as pd


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


class sm_net:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.count_times = 0

    def forward(self, X):
        self.count_times += 1
        # print(self.count_times)
        return softmax(torch.matmul(X.reshape(-1, self.W.shape[0]), self.W) + self.b)

    def get_train_times(self):
        return self.count_times


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


class Updator():
    """Updator = trainer 训练器,指定使用什么损失函数"""

    def __init__(self, net, batch_size, lr):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr

    def step(self):
        return d2l.sgd([self.net.W, self.net.b], self.lr, self.batch_size)


# @profile
def loadFashionMnistData(batch_size, resize=None):
    """下载FashionMnist数据集并加载到内存中

    :param batch_size:
    :param resize:
    :return:返回训练集和测试集的DataLoader
    """
    # 通过ToTenser()这个类 将图像数据从PIL类型转为浮点型的tensor类型,并除以255使得所有的像素数值均在0-1之间(归一化)  #需要下载将download改为True
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)
    print("数据集加载成功", len(mnist_train), len(mnist_test))  # 60000 ,10000

    num_workers = 4  # 设置读取图片的进程数量 小于cpu的核心数
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers))


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def softmax(x):
    """
    求得所有的x的softmax值
    :param x:计算的矩阵
    :return: 返回每个位置的比例
    """
    X_exp = torch.exp(x)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def cross_entropy(y_hat, y):
    """计算交叉熵"""
    # y_hat 64*10  y 64*1
    # y中存放了真实的标签下标
    # 计算交叉熵相当于对正确下标求-Log,越大越好
    #
    # test1 = y_hat[0, 1]
    # test2 = y_hat[0][1]
    x = y_hat[range(len(y_hat)), y]
    #
    return -torch.log(x)


def num_correct(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(torch.int).sum()


def net_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        # 判断net是不是nn的一个模块 如果是 将网络设置为评估模式 告诉其不要计算梯度了 只用计算准确率
        net.eval()

    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(num_correct(net.forward(X), y), y.numel())
    return float(metric[0]) * 100 / metric[1]


def save_params(net):
    pd.DataFrame(net.W.detach().numpy()).to_csv('netParams/W.CSV', index=False)  # 不保存列名
    pd.DataFrame(net.b.detach().numpy()).to_csv('netParams/b.CSV', index=False)  # 不保存列名
    print("写入成功")


def get_params(net, W_path='./netParams/W.CSV', b_path='./netParams/b.CSV'):
    net.W = torch.from_numpy(np.array(pd.read_csv(W_path))).type(torch.float)
    net.b = torch.from_numpy(np.array(pd.read_csv(b_path))).type(torch.float).flatten()
    return net


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式
    metric = Accumulator(3)  # 3个变量的累加器
    for X, y in train_iter:
        y_hat = net.forward(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.sum().backward()
            updater.step()
        metric.add(
            float(l.sum()),  # 损失函数的和
            num_correct(y_hat, y),  # 正确的数量
            y.numel()  # 总数量
        )
        loss_ave = metric[0] / metric[2]
        accuracy_ave = metric[1] / metric[2]
    return loss_ave, accuracy_ave


def train(net, train_iter, test_iter, loss, num_epochs, updater, save=True):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[.3, .9],
                        legend=['train_loss', 'train_acc', 'test_acc'])
    for epoch in range(num_epochs):
        print(('*' * 10 + str(epoch + 1) + '*' * 10).center(50))
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = net_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    if save:
        save_params(net)
    animator.fig.show()
    train_loss, train_acc = train_metrics
    # assert train_loss < .5, train_loss
    # assert 1 >= train_acc > .7, train_acc
    # assert 1 >= test_acc > 0.7, test_acc


def predict(net, test_iter, n=6):  # @save
    """预测标签"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net.forward(X).argmax(axis=1))
    correct_preds = [true == pred for true, pred in zip(trues, preds)]
    # titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    data = np.vstack((np.array(trues), np.array(preds), np.array(correct_preds))).T
    df1 = pd.DataFrame(data=data, columns=['真实', '预测', '是否相等'])
    print(df1)
    print(f"正确率:{float(sum(correct_preds) * 100 / len(correct_preds))}%")
    # d2l.show_images(
    #     X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# @profile
def main():
    batch_size = 64
    # 加载数据集
    train_iter, test_iter = loadFashionMnistData(batch_size)

    num_inputs = 784
    num_outputs = 10
    lr = 0.1

    # 初始化权重
    W = torch.normal(0, .01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    num_epochs = 1
    net = sm_net(W, b)
    updater = Updator(net, batch_size, lr)
    # 开始训练网络,训练的同时将参数保存到本地csv文件
    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # 预测
    net = get_params(net)
    predict(net, test_iter)
    del train_iter
    del test_iter


if __name__ == '__main__':
    main()
