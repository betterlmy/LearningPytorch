import torch
import torch.nn as nn
import lmy


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """nin块的设计"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


def nin():
    return nn.Sequential(
        nin_block(in_channels=1, out_channels=96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(in_channels=96, out_channels=256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(in_channels=256, out_channels=384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2), nn.Dropout(),
        nin_block(in_channels=384, out_channels=10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )


def main():
    net = nin()
    lr, num_epochs, batch_size = .05, 10, 128
    train_iter, test_iter = lmy.loadFashionMnistData(batch_size, '../lmy/data/', resize=224)
    devices = lmy.getGPU(contain_cpu=False)
    if len(devices) == 0:
        devices = [torch.device('cpu')]
    lmy.train_GPU(net, train_iter, test_iter, num_epochs, lr, devices=devices, num_devices=2)


if __name__ == '__main__':
    main()
