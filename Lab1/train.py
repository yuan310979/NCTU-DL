import nn.functional as F

from model import model
from utils.plot import show_result
from utils.generator import generate_linear, generate_XOR_easy

_model = model()

#  X, y = generate_linear(100)
X, y = generate_XOR_easy()
y_head = []

for i in range(100000):
    loss = 0.0
    for _X, _y in zip(X, y):
        y_pred = _model(_X, _y)
        loss += F.cross_entropy_loss(y_pred, _y)
        _model.backward()
        _model.step()
    if i % 5000 == 0:
        print(f"Epoch {i} loss : {loss[0]}")

for _X, _y in zip(X, y):
    y_head.append(_model(_X, _y) > 0.5)

show_result(X, y, y_head)
