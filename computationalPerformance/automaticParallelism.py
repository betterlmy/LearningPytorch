import torch
import sys

sys.path.append('../')
import lmy

devices, _ = lmy.getGPU(.4, contain_cpu=False)


def run(x):
    """简单计算"""
    return [x.mm(x) for _ in range(10)]


print(devices)
x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4001, 4001), device=devices[1])

# 进行数据预热
run(x_gpu1)
run(x_gpu2)
# torch.cuda.synchronize(devices[0])
# torch.cuda.synchronize(devices[1])
# 基准测试
# with lmy.Timer("GPU1"):
#     run(x_gpu1)
#     torch.cuda.synchronize(devices[0])
#
# with lmy.Timer("GPU2"):
#     run(x_gpu2)
#     torch.cuda.synchronize(devices[1])
# GPU1 has run for 0.0825s
# GPU2 has run for 0.0375s


# 我们知道PyTorch支持前后端分离,所以我们可以在不同的GPU上进行计算,而且并不需要对代码进行修改
with lmy.Timer("GPU1 & GPU2"):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()

