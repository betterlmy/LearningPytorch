import torch

import d2l
import lmy
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    """基础vgg块"""
    layers = []  # 初始化
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1  # 默认初始1通道输入
    out_channels = None
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(
            vgg_block(num_convs, in_channels, out_channels)
        )
        in_channels = out_channels
    return nn.Sequential(
        *conv_blocks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(),  # 7是因为最后的卷积层的输出是7*7
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
        nn.Linear(4096, 10)
    )


def main():
    conv_arch = (
        (1, 64), (1, 128), (2, 256)
        , (2, 512), (2, 512)  # 第一个元素表示卷积层的数量 第二个元素表示输出通道数
    )
    # net = vgg(conv_arch)
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    print(small_conv_arch)
    net = vgg(small_conv_arch)

    lr, num_epochs, batch_size = .05, 10, 128
    train_iter, test_iter = lmy.loadFashionMnistData(batch_size, '../lmy/data/', resize=224)
    devices = lmy.getGPU(contain_cpu=False)
    if len(devices) == 0:
        devices = [torch.device('cpu')]
    lmy.train_GPU(net, train_iter, test_iter, num_epochs, lr, devices=devices, num_devices=2)


if __name__ == '__main__':
    main()
