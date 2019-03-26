import nn
import nn.functional as F

class model():
    def __init__(self):
        self.hidden_layer = [nn.module.Linear(2,3), nn.module.Linear(3,3)]
        self.output_layer = nn.module.Linear(3,1)

        self.y = None
        self.y_pred = None

    def forward(self, x):
        for _h in self.hidden_layer:
            x = _h(x)
        self.y_pred = self.output_layer(x)
        return self.y_pred 

    def backward(self):
        back_item = self.output_layer.backward(F.derivative_cross_entropy_loss(self.y_pred, self.y))
        for _h in self.hidden_layer[::-1]:
            back_item = _h.backward(back_item)

    def step(self, lr=1e-2):
        self.output_layer.step(lr)
        for _h in self.hidden_layer:
            _h.step(lr)

    def tick(self, x, y):
        self.y = y
        return self.forward(x)

    __call__ = tick
