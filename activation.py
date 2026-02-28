from .base import Operation
from numpy import ndarray
import numpy as np

class Sigmoid(Operation):
    '''
    Sigmoid activation function.
    '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad

class Linear(Operation):
    '''
    "Identity" activation function
    '''

    def __init__(self) -> None:
        '''Pass'''        
        super().__init__()

    def _output(self) -> ndarray:
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Pass through'''
        return output_grad

class ReLU(Operation):
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> ndarray:
        return np.maximum(0, self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # Il gradiente è 1 se l'input era > 0, altrimenti è 0
        mask = self.input_ >= 0
        return output_grad * mask

class LeakyReLU(Operation):
    '''
    Leaky ReLU activation function.
    Lascia passare un piccolo gradiente (alpha) quando l'input è negativo.
    '''
    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def _output(self) -> np.ndarray:
        # np.maximum confronta elemento per elemento
        return np.maximum(self.alpha * self.input_, self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # Se l'input era >= 0, il gradiente è 1. Altrimenti è alpha (es. 0.2)
        mask = np.where(self.input_ >= 0, 1.0, self.alpha)
        return output_grad * mask

class Softmax(Operation):
    '''
    Softmax activation function.
    Converte un vettore di logit in una distribuzione di probabilità:
    ogni valore è in [0, 1] e la somma su tutte le classi è 1.
    Usata nell'output layer per classificazione multi-classe.
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> ndarray:
        # Sottrae il massimo per stabilità numerica (evita overflow con exp)
        shifted = self.input_ - np.max(self.input_, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # Jacobiano semplificato: si sfrutta il fatto che
        # dL/dx_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
        # Questa forma è corretta quando accoppiata con CategoricalCrossEntropy
        dot = np.sum(output_grad * self.output, axis=1, keepdims=True)
        return self.output * (output_grad - dot)
