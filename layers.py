# layers.py
import numpy as np
from numpy import ndarray
from .base import Layer, Operation
from .operations import WeightMultiply, BiasAdd
from .activation import Sigmoid

class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):
        super().__init__(neurons)
        self.activation = activation
        self.seed = None # Inizializzato a None come default

    def _setup_layer(self, input_: ndarray) -> None:
        if getattr(self, "seed", None):
            np.random.seed(self.seed)

        # Inizializzazione Xavier/Glorot
        in_dim = input_.shape[1]
        out_dim = self.neurons
        scale = np.sqrt(2.0 / (in_dim + out_dim))

        self.params = []
        self.params.append(np.random.standard_normal((in_dim, out_dim)) * scale) # Weights
        self.params.append(np.zeros((1, out_dim))) # Bias a zero

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
        return None
