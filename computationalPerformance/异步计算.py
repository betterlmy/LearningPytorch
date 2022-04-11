import numpy as np
import torch
import sys

sys.path.append('../')
import lmy

numpyTimer = lmy.Timer('Numpy Timer')

with numpyTimer:
    for i in range(30):
        a = np.random.normal(size=(1000, 1000))
        b = a @ a

devices, _ = lmy.getGPU(1)
device = devices[0]
# torchTimer1 = lmy.Timer('torch Timer1')
# with torchTimer1:
#     for i in range(30):
#         a = torch.randn((3000, 3000), device=device)
#         b = torch.mm(a, a)

torchTimer2 = lmy.Timer('torch Timer2')
with torchTimer2:
    print(device)
    for i in range(30):
        a = torch.randn((1000, 1000), device=device)
        b = torch.mm(a, a)
    # if 'cuda' in device.type:
    #     torch.cuda.synchronize(device)
