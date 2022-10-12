import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import time

from re import L
import numpy as np
from scipy.stats import norm

# 模型定义
class VAE(nn.Module):
    def __init__(self, input_dim, inter_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim),
            nn.Sigmoid()
        )

    @staticmethod
    def reparameter(mu, logvar):
        """
        返回重参数后的结果
        :rtype: double
        """
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        org_size = x.size()
        x = x.view(org_size[0], -1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameter(mu, logvar)
        recon_x = self.decoder(z).view(size=org_size)
        return recon_x, mu, logvar

# 定义损失函数
kl_loss = lambda mu,logvar : -0.5*torch.sum(1+logvar-mu.pow(2)-torch.exp(logvar))
recon_loss = lambda rencon_x,x:F.binary_cross_entropy(rencon_x,x,size_average=False)

# 定义训练参数
batch_size = 1024
transform = transforms.Compose([transforms.ToTensor()])
data_train = MNIST('data/MNIST',train=True,download=True,transform=transform)
data_test = MNIST('data/MNIST',train=False,download=True,transform=transform)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False,num_workers=0)

latent_dim = 2
input_dim = 28*28
inter_dim = 256
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
model = VAE(input_dim,inter_dim,latent_dim)
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=5e-3)

def train_VAE(num_epoch):
    best_loss = 1e9
    best_epoch = 0
    valid_losses = []
    train_losses = []
    for epoch in range(num_epoch):
        start_time = time.time()
        print(f"epoch {epoch+1} start")
        model.train()

        train_loss = 0.
        train_num = len(train_loader.dataset)

        for idx, (x, _) in enumerate(train_loader):
            batch = x.size(0) # 当前处理的batch中的图片数量,因为并不是每个batch都是batch_size
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)

            loss = recon+kl

            train_loss += loss.item()
            loss = loss / batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx %100 == 0:
                print(f"Train Loss:{loss:.3f},Recon Loss:{recon/batch:.3f},KL Loss:{kl/batch:.3f} in step {idx}")
            
        train_losses.append(train_loss/train_num)
        
        # 验证
        valid_loss = 0.
        valid_recon = 0.
        valid_kl = 0.
        valid_num = len(test_loader.dataset)
        model.eval()
        with torch.no_grad():
            for idx,(x,_) in enumerate(test_loader):
                x = x.to(device)
                recon_x ,mu,logvar = model(x)
                recon = recon_loss(recon_x ,x)
                kl = kl_loss(mu,logvar)
                
                loss = recon+kl
                valid_loss += loss.item()
                valid_recon += recon.item()
                valid_kl += kl.item()
            valid_losses.append(valid_loss/valid_num)
            
            print(f"Valid Loss:{loss:.3f},Recon Loss:{recon/valid_num:.3f},KL Loss:{kl/valid_num:.3f} in step {idx}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                
                torch.save(model,"model/best_VAE_MNIST_Model.pth")
                torch.save(model.state_dict(), 'model/best_VAE_MNIST_Para')
                print("适用于验证集的最佳模型已保存")
        print(f"epoch:{epoch} finished in {time.time()-start_time:.3f}s,best epoch = {best_epoch}")
        

    # len(train_losses),len(valid_losses)
    plt.plot(train_losses,label='Train')
    plt.plot(valid_losses,label='Valid')
    plt.legend(loc='upper right')
    plt.ylabel("loss(recon+kl)")
    plt.xlabel("epoch")
    plt.savefig("output/VAE_MNIST_Loss.png")


def valid_VAE(model=model):
    n = 20
    digit_size = 28 # 单个图像为28*28

    grid_x = norm.ppf(np.linspace(0.05,0.95,n))
    grid_y = norm.ppf(np.linspace(0.05,0.95,n))

    model.eval()
    figure = np.zeros((digit_size*n,digit_size*n)) # 单个图像为28*28,整个画布共20*20个图像
    for i,yi in enumerate(grid_y):
        for j,xi in enumerate(grid_x):
            z_sampled = torch.FloatTensor([xi,yi])
            with torch.no_grad():
                decoded = model.decoder(z_sampled)
                digit = decoded.view((digit_size,digit_size)) # 将图像转换为28*28
                figure[
                    i*digit_size:(i+1)*digit_size,
                    j*digit_size:(j+1)*digit_size
                    ]=digit

    plt.figure(figsize=(10,10))
    plt.imshow(figure,cmap="gray")      
    plt.axis("off")
    plt.savefig("output/VAE_MNIST_Decode.png")

import os

if __name__ == '__main__':
    # train_VAE(num_epoch=10)
    # valid_VAE()
    pass