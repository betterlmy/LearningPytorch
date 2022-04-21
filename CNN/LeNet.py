import torch
import torch.nn as nn
import sys

sys.path.append('../')
sys.path.append('../d2l.py')
import d2l

import lmy


net1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第一次采样
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第二次采样
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

net2 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 第一次采样
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 第二次采样
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_GPU(net, train_iter, test_iter, num_epochs, lr, timer=lmy.Timer(), device=torch.device('cpu')):
    """用GPU训练模型"""
    net.apply(init_weights)

    if "cuda" == device.type:
        net.to(device)
    timer.start()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = lmy.Animator(xlabel='epoch',
                            xlim=[1, num_epochs],
                            ylim=[.0, 1.0],
                            legend=['train_loss', 'train_acc', 'test_acc']
                            )
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        if timer.state == 'stopped':
            timer.start()
        metric = lmy.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            if timer.state == 'stopped':
                timer.start()
            optimizer.zero_grad()
            timer.stop()
            if 'cuda' in device.type:
                X, y = X.to(device), y.to(device)
            timer.start()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

        test_acc = lmy.evaluate_accuracy_gpu(net, test_iter, device, timer)
        animator.add(epoch + 1, (None, None, test_acc))
        timer.stop()
        print(f"loss:{train_l:.3f},train_acc:{train_acc:.3f},test_acc:{test_acc:.3f})")
        # print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {device}")


def main():
    batch_size = 256
    train_iter, test_iter = lmy.loadFashionMnistData(batch_size, "../lmy/data")

    lr = .9
    num_epochs = 10
    timer = lmy.Timer()
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    train_GPU(net1, train_iter, test_iter, num_epochs, lr, timer, device)
    print(f"{timer.sum():.1f} sec to train 1,000,000 samples of lenet on {device.type}")


if __name__ == '__main__':
    main()
