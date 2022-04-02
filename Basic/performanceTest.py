# GPU性能测试
import torch
import time
from torch import autograd

# GPU加速
print(torch.__version__)
ava = torch.cuda.is_available()
print(f"cuda{ava}")

a = torch.randn(40000, 1500)
b = torch.randn(1500, 40000)
print(a)
print(b)

if ava:
    device = torch.device('cuda:1')
    memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024

    memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    torch.cuda.empty_cache()
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
else:
    print("cpu")
    t0 = time.time()
    c = torch.matmul(a, b)
    t1 = time.time()

    print(a.device, t1 - t0, c.norm(2))