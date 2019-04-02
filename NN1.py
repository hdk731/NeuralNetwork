import numpy as np


class MLP:
    def __init__(self, input_units, hidden_units, output_units):
        self.W1 = np.random.randn(input_units, hidden_units)
        self.b1 = np.random.randn(1, hidden_units)
        self.W2 = np.random.randn(hidden_units, output_units)
        self.b2 = np.random.randn(1, output_units)

    # シグモイド関数
    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    # シグモイド関数の微分
    def sigmoid_derivative(self, x):
        y = self.sigmoid(x) * (1 - self.sigmoid(x))
        return y

    # トレーニング関数
    def train(self, X_train, y_train, epochs, learning_rate):
        for i in range(epochs):

            m = X_train.shape[0]

            # 順伝播・フォワードプロパゲーション・Forward-Propagation
            layer_z1 = np.dot(X_train, self.W1) + self.b1
            layer_a1 = self.sigmoid(layer_z1)
            layer_z2 = np.dot(layer_a1, self.W2)  + self.b2
            layer_a2 = self.sigmoid(layer_z2)

            # 目的関数・コスト関数・誤差関数
            cost = - np.sum(y_train * np.log(layer_a2) + (1 - y_train) * np.log(1 - layer_a2)) / m
            loss.append(cost)

            # 誤差逆伝播法・バックプロパゲーション・Back-Propagation
            dlayer_z2 = (layer_a2 - y_train)/m
            dW2 = np.dot(layer_a1.T, dlayer_z2)
            db2 = np.sum(dlayer_z2, axis=0 ,keepdims=True)

            dlayer_z1 = np.dot(dlayer_z2, self.W2.T) * self.sigmoid_derivative(layer_z1)
            dW1 = np.dot(X_train.T, dlayer_z1)
            db1 = np.sum(dlayer_z1, axis=0 ,keepdims=True)

            # パラメータ更新
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        return layer_a2


if __name__ == '__main__':
    # ユニットサイズ
    input_units, hidden_units, output_units = (2, 2, 1)
    # 入力データ
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # 教師データ
    y_train = np.array([[0], [1], [1], [0]])
    # エポック
    epochs = 30000
    # 学習率
    learning_rate = 0.1

    # コストを記録
    loss = []
    # インスタンスを生成
    mlp = MLP(input_units, hidden_units, output_units)
    # trainメソッドの呼び出し
    mlp.train(X_train, y_train, epochs, learning_rate)
