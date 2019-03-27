import numpy as np
import matplotlib.pyplot as plt

from nn import functional as F

cross_X = []
cross_y = []

mse_X = []
mse_y = []

for i in np.arange(0, 1, 0.01):
    cross_X.append(i)
    cross_y.append(F.cross_entropy_loss(i, 0))

    mse_X.append(i)
    mse_y.append(F.mse_loss(i, 0))

plt.title('Loss Function Comparison (y=0)')
plt.plot(cross_X, cross_y, 'b', label="Entropy Loss")
plt.plot(mse_X, mse_y, 'r', label="MSE")
plt.legend(loc='upper left')
plt.show()
