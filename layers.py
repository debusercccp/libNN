# layers.py
import numpy as np
from numpy import ndarray
from .base import Layer, Operation
from .operations import WeightMultiply, BiasAdd
from .activation import Sigmoid

class Dense(Layer):
    '''
    A fully connected (dense) layer which inherits from "Layer".
    Uses He initialization suitable for ReLU activations.
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):
        super().__init__(neurons)
        self.activation = activation
        self.seed = None  # Initialized to None as default

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Initialize layer parameters (weights and biases).
        Uses He initialization: variance = 2.0 / input_dim
        This is optimal for ReLU-like activations.
        
        Args:
            input_: Input batch with shape (batch_size, input_dim)
        '''
        # Check if seed was explicitly set (including 0!)
        if getattr(self, "seed", None) is not None:
            np.random.seed(self.seed)

        # He initialization (suitable for ReLU and variants)
        in_dim = input_.shape[1]
        out_dim = self.neurons
        scale = np.sqrt(2.0 / in_dim)  # He initialization

        self.params = []
        self.params.append(np.random.standard_normal((in_dim, out_dim)) * scale)  # Weights
        self.params.append(np.zeros((1, out_dim)))  # Bias (zero initialization is fine)

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
        return None
