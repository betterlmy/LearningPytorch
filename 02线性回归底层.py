import random
import torch


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


if __name__ == '__main__':
    # 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)  # 偏差是一个标量

    batch_size = 10
    num_epochs = 3
    net = linreg
    loss = squared_loss
    lr = .03

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # l.shape = batch_size * 1
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            # 下面的操作不需要计算梯度
            # 对所有的样本进行计算损失
            train_l = loss(net(features, w, b), labels)
            print(f"epoch{epoch + 1},loss{float(train_l.mean()):f}")

    print("w=", w)
    print("b=", b)
    print(f"w的误差:{true_w - w.reshape(true_w.shape)}")
    print(f"b的误差:{true_b - b}")
