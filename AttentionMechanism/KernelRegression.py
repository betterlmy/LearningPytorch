import torch

n_train = 50 # 数据集的大小

x_train,_ = torch.sort(torch.rand(n_train)*5) # 生成随机数据 范围是[0,5) 从小到大排序

def f(x):
    # 生成数据的函数
    return 2*torch.sin(x)+x**0.8

y_train = f(x_train) + torch.normal(0.0,0.5,(n_train,))# 生成数据
x_test = torch.arange(0,5,.1) # 测试数据
y_truth = f(x_test) # 测试数据的真实值
n_test