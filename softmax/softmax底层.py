import numpy as np
import torch
import torchvision
from memory_profiler import profile
from torch.utils import data
from torchvision import transforms
import lmy
import pandas as pd


class SoftMaxNet(lmy.Net):
    def __init__(self, num_inputs, num_outputs):
        W = torch.normal(0, .01, size=(num_inputs, num_outputs), requires_grad=True)
        b = torch.zeros(num_outputs, requires_grad=True)
        super().__init__()
        self.W = W
        self.b = b

    def forward(self, X):
        self.count_times += 1
        return softmax(torch.matmul(X.reshape(-1, self.W.shape[0]), self.W) + self.b)

    @property
    def params_names(self):
        params_names = ['W', 'b']
        params_values = (self.W, self.b)
        return zip(params_names, params_values)

    @property
    def params(self):
        return self.W, self.b


def loadFashionMnistData(batch_size, root="./data", resize=None):
    """下载FashionMnist数据集并加载到内存中

    :param root:
    :param batch_size:
    :param resize:
    :return:返回训练集和测试集的DataLoader
    """
    # 通过ToTenser()这个类 将图像数据从PIL类型转为浮点型的tensor类型,并除以255使得所有的像素数值均在0-1之间(归一化)  #需要下载将download改为True
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, download=False)
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


def num_correct(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(torch.int).sum()


def net_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        # 判断net是不是nn的一个模块 如果是 将网络设置为评估模式 告诉其不要计算梯度了 只用计算准确率
        net.eval()

    metric = lmy.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(num_correct(net.forward(X), y), y.numel())
    return float(metric[0]) * 100 / metric[1]


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式
    metric = lmy.Accumulator(3)  # 3个变量的累加器
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
        metric.add(
            float(l.sum()),  # 损失函数的和
            num_correct(y_hat, y),  # 正确的数量
            y.numel()  # 总数量
        )
        loss_ave = metric[0] / metric[2]
        accuracy_ave = metric[1] / metric[2]
    return loss_ave, accuracy_ave


def train(net, train_iter, test_iter, loss, num_epochs, updater, save=True):
    # animator = lmy.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[.3, .9],
    #                         legend=['train_loss', 'train_acc', 'test_acc'])
    for epoch in range(num_epochs):
        print(('*' * 10 + str(epoch + 1) + '*' * 10).center(50))
        train_metrics = train_epoch(net, train_iter, loss, updater)
        # test_acc = net_accuracy(net, test_iter)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    if save:
        lmy.save_params(net)
    # animator.fig.show()
    # train_loss, train_acc = train_metrics
    # assert train_loss < .5, train_loss
    # assert 1 >= train_acc > .7, train_acc
    # assert 1 >= test_acc > 0.7, test_acc


def predict(net, test_iter):  # @save
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
    batch_size = 256
    # 加载数据集
    train_iter, test_iter = loadFashionMnistData(batch_size)

    num_inputs = 784
    num_outputs = 10
    lr = 0.1
    num_epochs = 10
    net = SoftMaxNet(num_inputs, num_outputs)
    # updater = torch.optim.SGD(net.params, lr=lr)

    updater = lmy.SGD(net, lr, batch_size)
    # 开始训练网络,训练的同时将参数保存到本地csv文件
    train(net, train_iter, test_iter, lmy.cross_entropy, num_epochs, updater, True)
    # 预测
    # net = lmy.get_params(net)
    predict(net, test_iter)
    del train_iter
    del test_iter


if __name__ == '__main__':
    main()
