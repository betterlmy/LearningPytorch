import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath("./"))
import lmy
from thop import profile

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 10)
)

net1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1)
)


# X = torch.randn(1, 1, 224, 224, device='cuda:1')
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def main():

    # X = torch.randn(1, 3, 224, 224)
    # macs, params = profile(net1, (X,))
    #
    # print(macs, params)
    # # print(layer.__class__.__name__, 'output shape:\t', X.shape)
    batch_size = 128
    train_iter, test_iter = lmy.loadFashionMnistData(batch_size, "./lmy/data", resize=224)
    lr = .01
    num_epochs = 10
    timer = lmy.Timer("AlexNet")
    lmy.train_GPU(net, train_iter, test_iter, num_epochs, lr, timer)


if __name__ == "__main__":
    main()
