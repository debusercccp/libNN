import numpy as np
from numpy import ndarray
from .base import Loss

class MeanSquaredError(Loss):
    '''
    Mean Squared Error (MSE) loss.
    L = 1/N * sum((pred - target)^2)
    
    Use for: Regression tasks.
    Output activation: Linear()
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> float:
        '''Compute MSE: average squared differences'''
        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        '''Gradient: 2 * (pred - target) / N'''
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class BinaryCrossEntropy(Loss):
    '''
    Binary Cross-Entropy (BCE) loss.
    L = -1/N * sum(target * log(pred) + (1 - target) * log(1 - pred))
    
    Use for: Binary classification (two mutually exclusive classes).
    Output activation: Sigmoid()
    
    Predictions should be in (0, 1) range from Sigmoid output.
    '''
    def __init__(self, eps: float = 1e-9) -> None:
        '''
        Initialize BCE.
        
        Args:
            eps: Small epsilon to prevent log(0). Clips predictions to [eps, 1-eps].
        '''
        super().__init__()
        self.eps = eps

    def _output(self) -> float:
        '''Compute binary cross-entropy with numerical stability'''
        self.clipped_pred = np.clip(self.prediction, self.eps, 1 - self.eps)
        return -np.mean(self.target * np.log(self.clipped_pred) + 
                        (1 - self.target) * np.log(1 - self.clipped_pred))

    def _input_grad(self) -> ndarray:
        '''Gradient w.r.t. predictions: (pred - target) / (pred * (1 - pred)) / N'''
        grad = (self.clipped_pred - self.target) / (self.clipped_pred * (1 - self.clipped_pred))
        return grad / self.prediction.shape[0]


class DiceLoss(Loss):
    '''
    Dice Loss (F1 loss).
    L = 1 - (2 * |X ∩ Y| / (|X| + |Y|))
    
    Use for: Segmentation with class imbalance (especially few pixels of target class).
    Output activation: Sigmoid() (for pixel-wise binary)
    
    Heavily penalizes false negatives more than false positives.
    '''
    def __init__(self, smooth: float = 1.0) -> None:
        '''
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing constant to avoid division by zero.
        '''
        super().__init__()
        self.smooth = smooth

    def _output(self) -> float:
        '''Compute Dice coefficient and loss'''
        pred_flat = self.prediction.reshape(self.prediction.shape[0], -1)
        target_flat = self.target.reshape(self.target.shape[0], -1)
        intersection = np.sum(pred_flat * target_flat, axis=1)
        union = np.sum(pred_flat, axis=1) + np.sum(target_flat, axis=1)
        self.dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return np.mean(1 - self.dice_coeff)

    def _input_grad(self) -> ndarray:
        '''Gradient of Dice loss w.r.t. predictions'''
        pred_flat = self.prediction.reshape(self.prediction.shape[0], -1)
        target_flat = self.target.reshape(self.target.shape[0], -1)
        intersection = np.sum(pred_flat * target_flat, axis=1, keepdims=True)
        union = np.sum(pred_flat, axis=1, keepdims=True) + np.sum(target_flat, axis=1, keepdims=True)
        grad_flat = -2.0 * (target_flat * union - intersection) / ((union + self.smooth) ** 2)
        return grad_flat.reshape(self.prediction.shape) / self.prediction.shape[0]

class CategoricalCrossEntropy(Loss):
    '''
    Categorical Cross-Entropy (CCE) loss.
    L = -1/N * sum(target * log(pred))
    
    Use for: Multi-class classification (3+ mutually exclusive classes).
    Output activation: Softmax()
    
    Input format:
      - prediction: Softmax output, shape (n_samples, n_classes), values in (0, 1)
      - target: One-hot encoded, shape (n_samples, n_classes), values in {0, 1}
    
    IMPORTANT: Softmax + CategoricalCrossEntropy are paired.
    When combined, the gradient of the two operations simplifies to (pred - target).
    Using them separately with other combinations may produce unexpected results.
    
    Example:
        >>> y_pred = model.forward(X)  # output from Softmax
        >>> loss = net.loss.forward(y_pred, y_target_onehot)
        >>> grad = net.loss.backward()  # simplified by Softmax+CCE pairing
    '''
    def __init__(self, eps: float = 1e-9) -> None:
        '''
        Initialize Categorical Cross-Entropy.
        
        Args:
            eps: Small epsilon for numerical stability. Clips predictions to [eps, 1-eps].
        '''
        super().__init__()
        self.eps = eps

    def _output(self) -> float:
        '''Compute CCE with numerical stability via clipping'''
        self.clipped_pred = np.clip(self.prediction, self.eps, 1.0 - self.eps)
        return -np.mean(np.sum(self.target * np.log(self.clipped_pred), axis=1))

    def _input_grad(self) -> ndarray:
        '''
        Gradient of CCE w.r.t. predictions: -target / prediction / N
        
        Note: This is the direct gradient. When Softmax is applied before this loss,
        the backprop through Softmax further simplifies the total gradient to (pred - target).
        '''
        return -(self.target / self.clipped_pred) / self.prediction.shape[0]
