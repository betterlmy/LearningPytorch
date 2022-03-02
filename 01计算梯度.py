import torch

# 在计算y关于x的梯度之前,需要一个地方来存储梯度
X = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(X, X)
print(y)
y.backward()
# y.backward()
# 不能连续使用两次backward,因为X已经有了grad这个属性
print(X.grad[3])
X.grad = None
y = X.sum()
y.backward()

print(X.grad[3])


