import numpy as np

class Optimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self):
        for param, param_grad in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad

class SGDMomentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocities = None

    def step(self):
        if self.velocities is None:
            # Inizializziamo le velocità con zeri della stessa forma dei parametri
            self.velocities = [np.zeros_like(p) for p in self.net.params()]

        for i, (param, param_grad) in enumerate(zip(self.net.params(), self.net.param_grads())):
            # Aggiornamento della velocità: v = m * v + grad
            self.velocities[i] = self.momentum * self.velocities[i] + param_grad
            # Aggiornamento del parametro: p = p - lr * v
            param -= self.lr * self.velocities[i]
