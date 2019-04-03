import numpy as np


class MeanSquaredError(object):
    def forward(self, x, y):
        """
        損失関数
        :param x: <vector> 測定データ
        :param y: <vector> 教師データ
        :return: <vector> output
        """

        return np.square(y.reshape(-1) - x.reshape(-1)).mean()

    def backward(self, x, y):
        """
        損失関数
        :param x: <vector> 測定データ
        :param y: <vector> 教師データ
        :return: <vector>
        """

        return 2 * (y.reshape(-1) - x.reshape(-1)).mean()
