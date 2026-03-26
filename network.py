import copy
import typing
import pickle
from typing import Callable, List
import numpy as np
from numpy import ndarray

from .base import Layer, Loss
from .utils import assert_same_shape, permute_data

# --- Funzioni di Utility Matematiche ---

def square(x: ndarray) -> ndarray:
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x)) 

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          diff: float = 0.001) -> ndarray:
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)

# --- Classi Principali della Rete ---

class NeuralNetwork:
    '''La classe base per una rete neurale sequenziale.'''
    def __init__(self, layers: List[Layer], loss: Loss, seed: int = None):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        self.train = True 
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)        

    def forward(self, x_batch: ndarray) -> ndarray:
        '''Passa i dati attraverso i layer in sequenza.'''
        x_out = x_batch
        for layer in self.layers:
            # Impostiamo lo stato di training (fondamentale per Dropout)
            if hasattr(layer, 'is_training'):
                layer.is_training = self.train
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''Propaga il gradiente all'indietro.'''
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        '''Esegue un intero step di training su un batch.'''
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss
    
    def params(self):
        '''Genera i parametri dei layer, saltando quelli che non ne hanno (es. Dropout).'''
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params is not None:
                yield from layer.params

    def param_grads(self):
        '''Genera i gradienti dei layer, saltando quelli senza parametri.'''
        for layer in self.layers:
            if hasattr(layer, 'param_grads') and layer.param_grads is not None:
                yield from layer.param_grads 

    def save_model(self, filename="model.pkl"):
        """Salva l'intera struttura del modello e i pesi."""
        model_data = {
            'weights': self.weights, # Lista di matrici numpy
            'biases': self.biases,   # Lista di array numpy
            'topology': self.topology # Es: [10, 32, 1]
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f" Modello salvato con successo in: {filename}")

    @staticmethod
    def load_model(filename):
        """Carica un modello salvato e restituisce una nuova istanza di NN."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Impossibile trovare il file {filename}")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Creiamo una nuova istanza con la topologia salvata
        # Nota: adatta il nome della classe se diverso
        nn = NeuralNetwork(data['topology']) 
        nn.weights = data['weights']
        nn.biases = data['biases']
        print(f" Modello caricato da {filename}. Pronto per le predizioni!")
        return nn

# --- Trainer Class ---

class Trainer(object):
    '''Classe per gestire l'addestramento della rete.'''
    def __init__(self, net: NeuralNetwork, optim) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)
        
    def generate_batches(self, X: ndarray, y: ndarray, size: int = 32):
        assert X.shape[0] == y.shape[0], \
        f"Features ha {X.shape[0]} samples, target ne ha {y.shape[0]}"
        
        N = X.shape[0]
        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]
            yield X_batch, y_batch

    def _evaluate(self, X_test: ndarray, y_test: ndarray) -> float:
        '''Valuta il modello disattivando il Dropout.'''
        self.net.train = False 
        test_preds = self.net.forward(X_test)
        loss_val = self.net.loss.forward(test_preds, y_test)
        self.net.train = True
        return loss_val
            
    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int=100, eval_every: int=10,
            batch_size: int=32, seed: int = 1, restart: bool = True) -> None:
        
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        for e in range(epochs):
            # Clonazione per eventuale ripristino (Early Stopping)
            if (e+1) % eval_every == 0:
                last_model = copy.deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e+1) % eval_every == 0:
                loss = self._evaluate(X_test, y_test)
                if loss < self.best_loss:
                    print(f"Epoch {e+1}: Validation Loss = {loss:.4f}")
                    self.best_loss = loss
                else:
                    print(f"Loss aumentata a {loss:.4f}. Ripristino miglior modello.")
                    self.net = last_model
                    # Ricolleghiamo l'ottimizzatore alla copia corretta del modello
                    setattr(self.optim, 'net', self.net)
                    break
