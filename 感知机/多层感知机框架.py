import torch
import torch.nn as nn
import lmy
from softmax import softmax底层 as softmax


def main():
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    net.apply(lmy.init_weights)
    batch_size, lr, num_epochs = 256, .1, 10
    updater = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = softmax.loadFashionMnistData(batch_size, "../softmax/data")
    softmax.train(net, train_iter, test_iter, loss, num_epochs, updater)
    softmax.predict(net, test_iter)


if __name__ == '__main__':
    main()
