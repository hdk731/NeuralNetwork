import numpy as np


class Network(object):
    def __init__(self, n_i, n_o, lr):
        self.weight = np.random.randn(n_o, n_i)
        self.inputs = None
        self.grad = np.zeros((n_i, n_o))
        self.lr = lr

    # 順伝播 - Forward-Propagation
    def forward(self, x):
        # 入力値を保持
        self.inputs = x.reshape(-1, 1)
        # 入力値に重み付与
        return np.dot(self.weight, self.inputs)

    # 誤差逆伝播 - Back-Propagation
    def backward(self, dx):
        # 重み(w)に関する微分計算
        self.grad = np.dot(self.inputs.reshape(-1, 1), dx.reshape(1, -1)).reshape(self.weight.shape)
        # 勾配(x)に関する微分計算
        return np.dot(dx.reshape(1, -1), self.weight)

    def update(self):
        self.weight -= self.grad * self.lr


if __name__ == '__main__':
    fc = Network(10, 2, 0.1)
    x = np.random.randn(10, 1)
    fc.forward(x)

    # 前から来る微分値
    grad = np.random.randn(1, 2)
    fc.backward(grad)
    fc.update()
