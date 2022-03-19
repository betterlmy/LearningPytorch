import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import d2l
from Timer import Timer
from IPython import display


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


def loadFashionMnistData(batch_size, resize=None):
    """
    下载FashionMnist数据集并加载到内存中
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
    return -torch.log(y_hat[range(len(y_hat)), y])


def num_correct(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(torch.int).sum()


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


def net_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        # 判断net是不是nn的一个模块 如果是 将网络设置为评估模式 告诉其不要计算梯度了 只用计算准确率
        net.eval()

    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(num_correct(net.count(X), y), y.numel())
    return float(metric[0]) * 100 / metric[1]


class sm_net:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.count_times = 0

    def count(self, X):
        self.count_times += 1
        print(self.count_times)
        return softmax(torch.matmul(X.reshape(-1, self.W.shape[0]), self.W) + self.b)


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net.count(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l.sum()) * len(y),
                num_correct(y_hat, y),
                y.size().numel()
            )
        else:
            l.sum().backward()
            updater(net, X.shape[0])
            metric.add(
                float(l.sum()) * len(y),
                num_correct(y_hat, y),
                y.size().numel()
            )
        return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[.3, .9],
                        legend=['train_loss', 'train_acc', 'test_acc'])
    for epoch in range(num_epochs):
        print(('*' * 10 + str(epoch + 1) + '*' * 10).center(50))
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = net_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    assert train_loss < .5, train_loss
    assert 1 >= train_acc > .7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


def updater(net, batch_size):
    return d2l.sgd([net.W, net.b], lr, batch_size)


def predict(net, test_iter, n=6):  # @save
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net.count(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':

    batch_size = 64
    train_iter, test_iter = loadFashionMnistData(batch_size)

    num_inputs = 784
    num_outputs = 10
    lr = 0.1

    # 初始化权重
    W = torch.normal(0, .01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    num_epochs = 10
    net = sm_net(W, b)
    for X, y in train_iter:
        net.count(X)
    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict(net, test_iter)
