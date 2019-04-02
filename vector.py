import numpy as np
import nn

if __name__ == '__main__':
    fc = nn(10, 2, 0.1)
    x = np.random.randn(10, 1)
    fc.forward(x)
    grad = np.random.randn(1, 2)  # 前から来る微分値
    fc.backward(grad)
    fc.update()
