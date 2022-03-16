import math
import numpy as np
import torch
import d2l
import sys
from Timer import Timer

n = int(1e4)
a = torch.ones(n)
b = torch.ones(n)

c = torch.zeros(n)

# 通过for循环计算
timer1 = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f"{timer1.stop():.5f}")

# 通过矢量运算计算
timer1.start()
d = a + b
print(f"{timer1.stop():.5f}")
print(c == d)
