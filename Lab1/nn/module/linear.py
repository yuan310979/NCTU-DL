import numpy as np
from .. import functional as F

class Linear:
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = np.random.rand(out_feature, in_feature)

        self.x = None
        self.delta = None

    def forward(self, x):
        self.x = np.array(x)
        self.z = F.linear(self.weight, self.x.T)
        return F.sigmoid(self.z)

    def backward(self, x):
        self.delta = np.multiply(x, F.derivative_sigmoid(F.sigmoid(self.z)))
        return np.dot(self.weight.T, self.delta)

    def step(self, lr=1e-3):
        delta = self.delta.reshape(-1, 1)
        x = self.x.reshape(-1, 1) 
        self.weight = self.weight - lr * np.dot(delta, x.T)

    def tick(self, x):
        return self.forward(x)

    __call__ = tick
