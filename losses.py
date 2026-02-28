import numpy as np
from numpy import ndarray
from .base import Loss

class MeanSquaredError(Loss):
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> float:
        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class BinaryCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps

    def _output(self) -> float:
        self.clipped_pred = np.clip(self.prediction, self.eps, 1 - self.eps)
        return -np.mean(self.target * np.log(self.clipped_pred) + 
                        (1 - self.target) * np.log(1 - self.clipped_pred))

    def _input_grad(self) -> ndarray:
        grad = (self.clipped_pred - self.target) / (self.clipped_pred * (1 - self.clipped_pred))
        return grad / self.prediction.shape[0]


class DiceLoss(Loss):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def _output(self) -> float:
        pred_flat = self.prediction.reshape(self.prediction.shape[0], -1)
        target_flat = self.target.reshape(self.target.shape[0], -1)
        intersection = np.sum(pred_flat * target_flat, axis=1)
        union = np.sum(pred_flat, axis=1) + np.sum(target_flat, axis=1)
        self.dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return np.mean(1 - self.dice_coeff)

    def _input_grad(self) -> ndarray:
        pred_flat = self.prediction.reshape(self.prediction.shape[0], -1)
        target_flat = self.target.reshape(self.target.shape[0], -1)
        intersection = np.sum(pred_flat * target_flat, axis=1, keepdims=True)
        union = np.sum(pred_flat, axis=1, keepdims=True) + np.sum(target_flat, axis=1, keepdims=True)
        grad_flat = -2.0 * (target_flat * union - intersection) / ((union + self.smooth) ** 2)
        return grad_flat.reshape(self.prediction.shape) / self.prediction.shape[0]

class CategoricalCrossEntropy(Loss):
    '''
    Categorical Cross-Entropy loss per classificazione multi-classe.
    Si aspetta:
      - prediction : output Softmax, shape (n_samples, n_classes), valori in (0, 1)
      - target     : one-hot encoded,  shape (n_samples, n_classes), valori in {0, 1}

    Formula: L = -1/N * sum( target * log(prediction) )
    '''
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps  # piccolo valore per evitare log(0)

    def _output(self) -> float:
        # Clip per stabilità numerica
        self.clipped_pred = np.clip(self.prediction, self.eps, 1.0 - self.eps)
        return -np.mean(np.sum(self.target * np.log(self.clipped_pred), axis=1))

    def _input_grad(self) -> ndarray:
        # Gradiente di CCE rispetto all'output Softmax:
        # dL/dp_i = -target_i / prediction_i  (scalato per il batch)
        # Nota: quando Softmax e CCE sono accoppiate, il gradiente combinato
        # semplifica a (prediction - target) / n — ma qui gestiamo solo la loss,
        # il backward della Softmax fa il resto.
        return -(self.target / self.clipped_pred) / self.prediction.shape[0]
