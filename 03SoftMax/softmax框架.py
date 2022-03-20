import torch
from torch import nn
import d2l
import softmax底层


def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, std=.1)


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = softmax底层.loadFashionMnistData(batch_size)
    # torch不会隐式地调整输入的形状 例如从28*28转至1*784 因此 我们需要在线性层前通过定义展平层 flatten()来调整网络的输入形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    loss = nn.CrossEntropyLoss()
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=.1)
    epochs = 1
    softmax底层.train(net, train_iter, test_iter, loss, epochs, trainer, save=False)
    for key in net.state_dict().keys():
        print(f'{key},{net[1].weight}')
