# libNN/__init__.py

# Struttura principale
from .network import NeuralNetwork, Trainer

# Funzioni di Perdita (Loss)
from .losses import MeanSquaredError, BinaryCrossEntropy, DiceLoss, CategoricalCrossEntropy

# Layer e componenti
from .layers import Dense
from .activation import ReLU, LeakyReLU, Sigmoid, Linear, Softmax
from .operations import Dropout

# Ottimizzatori
from .optimizers import SGD, SGDMomentum

# Utility
from .utils import compute_accuracy, compute_f1_score, mae, rmse, to_one_hot
