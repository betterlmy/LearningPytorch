import torch
import torch.nn as nn
import lmy
from softmax import softmax底层 as softmax


def init_params(num_inputs, num_hiddens, num_outputs):
    W1 = torch.randn(num_inputs, num_hiddens, requires_grad=True)
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    W2 = torch.randn(num_hiddens, num_outputs, requires_grad=True)
    b2 = torch.zeros(num_outputs, requires_grad=True)
    return W1, b1, W2, b2


class PerceptronNet(lmy.Net):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.W1, self.b1, self.W2, self.b2 = init_params(num_inputs, num_hiddens, num_outputs)

    def forward(self, X):
        self.count_times += 1
        X = X.reshape((-1, self.num_inputs))
        # 隐含层计算
        X = X @ self.W1 + self.b1

        # 激活函数计算
        H = lmy.relu(X)

        # 结果层计算
        return H @ self.W2 + self.b2

    @property
    def params_names(self):
        params_names = ['W1', 'b1', 'W2', 'b2']
        params_values = (self.W1, self.b1, self.W2, self.b2)
        return zip(params_names, params_values)


def main():
    batch_size = 256
    num_inputs, num_outputs = 784, 10
    num_hiddens = 256
    num_epochs = 10
    just_predict = True

    train_iter, test_iter = softmax.loadFashionMnistData(batch_size, '../softmax/data')
    loss = lmy.cross_entropy

    net = PerceptronNet(num_inputs, num_hiddens, num_outputs)

    updator = lmy.SGD(net, .05, batch_size)

    if not just_predict:
        softmax.train(net, train_iter, test_iter, loss, num_epochs, updator, save=True)
    softmax.predict(net, test_iter)

    del train_iter
    del test_iter

    pass


if __name__ == '__main__':
    main()
