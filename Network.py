import numpy as np


class Network(object):
    def __init__(self, suffix_col, suffix_row, lr):
        self.weight = np.random.randn(suffix_row, suffix_col)
        self.inputs = None
        self.grad = np.zeros((suffix_col, suffix_row))
        self.lr = lr

    def forward(self, x):
        """
        順伝播 - Forward-Propagation
        :param x: <vector>
        :return: <vector> output
        """

        # 入力値を保持
        self.inputs = x.reshape(-1, 1)
        # 入力値に重み付与
        return np.dot(self.weight, self.inputs)

    def backward(self, dx):
        """
        誤差逆伝播 - Back-Propagation
        :param dx: <vector> l+1層の微分値
        :return: <vector>
        """

        # 重み(w)に関する微分計算
        self.grad = np.dot(self.inputs.reshape(-1, 1), dx.reshape(1, -1)).reshape(self.weight.shape)
        # 誤差(x)に関する微分計算
        return np.dot(dx.reshape(1, -1), self.weight)

    def update(self):
        print(self.weight)
        self.weight -= self.grad * self.lr
        print(self.weight)

