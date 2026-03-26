import numpy as np

class Optimizer:
    '''
    Base class for optimizers.
    Optimizers update network parameters based on their gradients.
    '''
    def __init__(self, lr: float = 0.01):
        '''
        Initialize Optimizer.
        
        Args:
            lr: Learning rate (default 0.01). Controls magnitude of parameter updates.
        '''
        assert lr > 0, f"Learning rate must be positive, got {lr}"
        self.lr = lr

    def step(self) -> None:
        '''
        Perform one optimization step to update all parameters.
        Must be called after backward() to apply accumulated gradients.
        
        Subclasses must override this method.
        '''
        raise NotImplementedError()

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent (SGD) optimizer.
    Simple parameter update: param = param - lr * gradient
    
    Good for: Basic training, often requires careful learning rate tuning.
    '''
    def __init__(self, lr: float = 0.01):
        '''
        Initialize SGD.
        
        Args:
            lr: Learning rate (default 0.01)
        '''
        super().__init__(lr)

    def step(self):
        '''
        Update all network parameters by moving them in direction opposite to gradient.
        Called after each batch.
        '''
        for param, param_grad in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad

class SGDMomentum(Optimizer):
    '''
    SGD with Momentum optimizer.
    Maintains velocity (momentum) of parameter updates to accelerate convergence
    and reduce oscillations.
    
    Update rule:
        v = momentum * v + gradient
        param = param - lr * v
    
    Good for: Faster convergence, smoother training curves, less sensitive to learning rate.
    '''
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        '''
        Initialize SGDMomentum.
        
        Args:
            lr: Learning rate (default 0.01)
            momentum: Momentum coefficient (default 0.9). Higher = more "inertia".
                      Typical range: [0.8, 0.99]
        '''
        super().__init__(lr)
        assert 0 <= momentum < 1, f"Momentum must be in [0, 1), got {momentum}"
        self.momentum = momentum
        self.velocities = None  # Will be initialized on first step()

    def step(self):
        '''
        Update parameters using momentum-accelerated gradients.
        Velocities are initialized on first call and then updated each step.
        
        The velocity accumulates gradients over time, allowing the optimizer
        to "roll downhill" faster in consistent directions.
        '''
        if self.velocities is None:
            # First step: initialize velocities with zeros (same shape as parameters)
            self.velocities = [np.zeros_like(p) for p in self.net.params()]

        for i, (param, param_grad) in enumerate(zip(self.net.params(), self.net.param_grads())):
            # Update velocity: accumulate gradient with momentum
            self.velocities[i] = self.momentum * self.velocities[i] + param_grad
            # Update parameter: move in direction of accumulated momentum
            param -= self.lr * self.velocities[i]
