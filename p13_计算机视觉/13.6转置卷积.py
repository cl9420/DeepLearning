import torch
from torch import nn
from d2l import torch as d2l
import cv2

def trans_conv(X, k):
    h, w = k.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * k
    return Y


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
X = X.reshape(1, 1, 2, 2)
K = K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, 64, 32, 16, bias=False)
# tconv.weight.data = K
print(tconv(X))
# print(trans_conv(X, K))

