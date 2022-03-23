import numpy as np
import torch
import torch.nn as nn

# 构造一组数据集
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
# 构造y = 2x+1
y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class linearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


input_dim = 1
output_dim = 1

model = linearRegressionModel(1, 1)
epoches = 200
lr = .01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss = nn.MSELoss()
inputs = torch.tensor(x_train)
labels = torch.tensor(y_train)

for epoch in range(epoches):
    epoch += 1
    optimizer.zero_grad()
    outputs = model(inputs)
    ls = loss(outputs, labels)
    ls.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch:{epoch} loss:{ls:f}")
