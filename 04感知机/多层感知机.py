import torch
import torch.nn as nn
import d2l
from softmax import softmax底层 as softmax

batch_size = 256
train_iter, test_iter = softmax.loadFashionMnistData(batch_size)
