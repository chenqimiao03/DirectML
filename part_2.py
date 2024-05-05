import torch
import torch.nn as nn


def corr2d(X, K):
    """计算二维相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    """实现二维卷积层"""

    def __init__(self, kernel_size):
        super().__init___()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def comp_conv2d(conv2d, X):
    # 插入 batch_size 和 通道数
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def corr2d_multi_in(X, K):
    """实现多输入通道互相关运算"""
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_out(X, K):
    """实现多输出通道互相关运算"""
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_1x1(X, K):
    """1x1卷积"""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


def pool2d(X, pool_size, mode="max"):
    """实现池化层（没有步长和填充）"""
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            else:
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
