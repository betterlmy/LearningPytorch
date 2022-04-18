import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils import data
import os
import sys

sys.path.append('../')
sys.path.append('../d2l.py')
import lmy

if not os.path.exists('./img'):
    os.mkdir('./img')

if not os.path.exists('./model'):
    os.mkdir('./model')


def to_img(x):
    """将tensor转为图片格式"""
    x = (x + 1) * .5
    x = x.clamp(0, 1)  # clamp 夹紧，将x的值限制在0-1之间
    x = x.view(-1, 1, 28, 28)  # view()函数的作用，将多行的tensor转换为多个二维的图片
    return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(.2),
            nn.Linear(256, 256), nn.LeakyReLU(.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

    def __call__(self, x):
        return self.forward(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维 输入是100维
            nn.ReLU(),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        return self.generator(x)

    def __call__(self, x):
        return self.forward(x)


def main():
    batch_size = 128
    num_epochs = 100
    z_dimension = 100
    # 图像预处理
    img_transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为tensor
        transforms.Normalize([0.5], [0.5])  # 归一化
    ])
    # 读取数据
    mnist_train = FashionMNIST(root='../lmy/data', train=True, transform=img_transform, download=False)
    bags_data = []
    for i, (img, label) in enumerate(mnist_train):
        if label == 8:
            bags_data.append(img)
    train_iter = data.DataLoader(bags_data, batch_size, shuffle=False, num_workers=4)
    D = Discriminator()
    G = Generator()
    devices = lmy.getGPU(contain_cpu=True)
    cuda_available = False
    if 'cuda' in devices[0].type:
        cuda_available = True

    if cuda_available:
        D = D.to(devices[0])
        G = G.to(devices[0])

    criterion = nn.BCELoss()  # 二进制交叉熵 因为结果只有True和False
    g_optimizer = torch.optim.Adam(G.parameters(), lr=.001)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=.001)

    for epoch in range(num_epochs):
        for i, train_imgs in enumerate(train_iter):
            num_imgs = len(train_imgs)  # = batch_size
            # ====================训练判别器D==================#
            # 判别器的训练分为两个部分：1 真实图像判别为真 2 生成图像判别为假

            train_imgs = train_imgs.reshape(num_imgs, -1)  # 拉平 将一个batch所有的图片放到一个tensor中
            # print(train_imgs.shape) # torch.Size([128, 784])

            real_img = Variable(train_imgs)  # tensor转变为Vairable类型的变量
            real_label = Variable(torch.ones(num_imgs))  # 定义真实的图片label为1
            fake_label = Variable(torch.zeros(num_imgs))  # 定义虚假的图片label为0
            if cuda_available:
                real_img, real_label, fake_label = real_img.to(devices[0]), real_label.to(devices[0]), fake_label.to(
                    devices[0])

            # 计算真实图片的损失
            d_real_out = D(real_img).squeeze()  # 真实图片的输出
            d_real_loss = criterion(d_real_out, real_label)
            real_scores = d_real_out

            # 计算假图片的损失
            z = Variable(torch.randn(num_imgs, z_dimension))
            if cuda_available:
                z = z.to(devices[0])

            fake_img = G(z).detach()  # 生成假图片
            fake_out = D(fake_img).squeeze()  # 使用判别器对假图片进行判断
            d_fake_loss = criterion(fake_out, fake_label)
            fake_scores = fake_out
            d_loss = d_real_loss + d_fake_loss  # 损失包括真损失和假损失
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ====================训练生成器G==================#
            # 目的：希望生成的假图片被判别器D判断为True
            # 生成器的训练分两个部分：1 生成假图片 2 判别器对假图片判别为真
            # 过程中 将判别器参数固定，将假的图片传入判别器的结果与真实的label对应

            z = Variable(torch.randn(num_imgs, z_dimension))
            if cuda_available:
                z = z.to(devices[0])
            fake_img = G(z).detach()
            output = D(fake_img).squeeze()
            g_loss = criterion(output, real_label)
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()
            g_optimizer.step()
            # 打印中间的损失
            if (i + 1) % 20 == 0:
                print(
                    f'Epoch[{epoch}/{num_epochs}],d_loss:{d_loss.data.item():.6f},g_loss:{g_loss.data.item():.6f} ,D real: {real_scores.data.mean():.6f},D fake: {fake_scores.data.mean():.6f}')  # 打印的是真实图片的损失均值
            if epoch == 0:
                real_images = to_img(real_img.cpu().data)
                torchvision.utils.save_image(real_images, './img/real_images.png')

        fake_images = to_img(fake_img.data)
        torchvision.utils.save_image(fake_images, f'./img/fake_images-{epoch + 1}.png')
    # 保存模型
    torch.save(G.state_dict(), './model/generator.pth')
    torch.save(D.state_dict(), './model/discriminator.pth')
