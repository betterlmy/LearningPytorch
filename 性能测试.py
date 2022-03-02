# GPU性能测试
import torch
import time
from torch import autograd

#GPU加速
print(torch.__version__)
ava= torch.cuda.is_available()
print(ava)

a = torch.randn(30000, 1000)
b = torch.randn(1000, 30000)
print(a)
print(b)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()

print(a.device, t1 - t0, c.norm(2))
if ava:
    device = torch.device('cuda')
    print(device)
    a = a.to(device)
    b = b.to(device)

    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()
    print(a.device, t2 - t0, c.norm(2))

    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()

    print(a.device, t2 - t0, c.norm(2))