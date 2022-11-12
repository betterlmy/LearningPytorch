import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from utils import *
import time
mkdir("./img")

device = get_device("cpu")
print(device)
batch_size = 128
num_epoch = 100
z_dimension = 100

img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

# 没有测试集
mnist = datasets.MNIST(
    root="./data/",train=True,transform=img_transform,download=True
)

dataloader = torch.utils.data.DataLoader(dataset = mnist,batch_size=batch_size,shuffle=True)


# 传统的GAN仅使用MLP实现判别器和生成器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(28*28, 256),nn.LeakyReLU(.1),
            nn.Linear(256,256),nn.LeakyReLU(.1),
            nn.Linear(256,1),
            nn.Sigmoid() # 二分类用Sigmoid 多分类用softmax
        )

    def forward(self,x):
        return self.dis(x)

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100,256),nn.ReLU(True),
            nn.Linear(256,256),nn.ReLU(True),
            nn.Linear(256,784),
            nn.Tanh() # Tanh使得生成的图片像素值在-1到1之间
        )
    def forward(self, x):
        return self.gen(x)

# 实例化对象
D = Discriminator()
G = Generator()
D = D.to(device)
G = G.to(device)

### 鉴别器训练
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr=.0003)
g_optimizer = torch.optim.Adam(G.parameters(),lr=.0003)

for epoch in range(num_epoch):
    start_time = time.time()


    for i ,(img,_) in enumerate(dataloader):
        num_img = img.size(0) # batch中图片的数量
        for a in range(10):
            # 训练判别器
            img = img.view(num_img,-1) # 将图片展开成28*28=784的向量
            real_img = img.to(device)
            real_label = torch.ones(num_img).to(device)
            fake_label = torch.zeros(num_img).to(device)

            real_out = D(real_img).squeeze(1)
            d_loss_real = criterion(real_out,real_label)
            real_scores = real_out


            z = torch.randn(num_img,z_dimension).to(device)
            fake_img = G(z)
            fake_out = D(fake_img).squeeze(1)
            d_loss_fake = criterion(fake_out,fake_label)
            fake_scores = fake_out

            d_loss = d_loss_real+d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


        # 训练生成器
        z = torch.randn(num_img,z_dimension).to(device)
        fake_img = G(z)
        output = D(fake_img).squeeze(1) #?
        g_loss =criterion(output,real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # 打印中间损失
        if (i+1) %100 ==0:
            print(f"epoch {epoch}/{num_epoch},d_loss:{d_loss.item():.6f},"
            f"g_loss:{g_loss.item():.6f},D real:{round(torch.mean(real_scores).item(),2)},D fake:{torch.mean(fake_scores).item()}")

        # 保存图片
        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, f'./img/fake_images-{epoch + 1}.png')
    print(f'{time.time()-start_time}s/epoch')

# 模型保存
torch.save(G.state_dict(), './GAN_G.pth')
torch.save(D.state_dict(), './GAN_D.pth')