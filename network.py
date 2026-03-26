import os
import copy
import typing
import pickle
from typing import Callable, List
import numpy as np
from numpy import ndarray

from .base import Layer, Loss
from .utils import assert_same_shape, permute_data

# --- Classi Principali della Rete ---

class NeuralNetwork:
    '''La classe base per una rete neurale sequenziale.'''
    def __init__(self, layers: List[Layer], loss: Loss, seed: int = None):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        self.train = True 
        self.best_loss = float('inf') # Per il tracking del miglior modello
        
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)        

    def forward(self, x_batch: ndarray) -> ndarray:
        x_out = x_batch
        for layer in self.layers:
            if hasattr(layer, 'is_training'):
                layer.is_training = self.train
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    # --- NUOVE FUNZIONALITÀ DI SALVATAGGIO ---

    def save_model(self, filename="model.pkl", is_best=False):
        """
        Salva l'intera struttura dei layer e i loro parametri.
        Se is_best è True, stampa un messaggio speciale.
        """
        with open(filename, 'wb') as f:
            # Salviamo l'intera lista di layer, la loss e il seed
            pickle.dump({
                'layers': self.layers,
                'loss': self.loss,
                'seed': self.seed,
                'best_loss': self.best_loss
            }, f)
        
        status = " (Miglior modello finora! 🏆)" if is_best else ""
        print(f" Modello salvato in: {filename}{status}")

    @staticmethod
    def load_model(filename):
        """Carica un modello e ricostruisce l'istanza NeuralNetwork."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} non trovato.")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Ricostruiamo la classe usando i dati salvati
        nn = NeuralNetwork(layers=data['layers'], loss=data['loss'], seed=data.get('seed'))
        nn.best_loss = data.get('best_loss', float('inf'))
        
        print(f" Modello caricato con successo da {filename}.")
        return nn

    def check_and_save(self, current_loss: float, filename="best_model.pkl"):
        """Utility per l'auto-checkpoint: salva solo se la loss è migliorata."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_model(filename, is_best=True)

    # --- METODI PER I PARAMETRI ---

    def params(self):
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params is not None:
                yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            if hasattr(layer, 'param_grads') and layer.param_grads is not None:
                yield from layer.param_grads

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
            epochs: int = 100, eval_every: int = 10,
            batch_size: int = 32, seed: int = 1,
            restart: bool = True, patience: int = 5) -> None:

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        best_model = None
        patience_counter = 0

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e + 1) % eval_every == 0:
                loss = self._evaluate(X_test, y_test)

                if loss < self.best_loss:
                    self.best_loss = loss
                    best_model = copy.deepcopy(self.net)
                    patience_counter = 0
                    print(f"Epoch {e+1}: Validation Loss = {loss:.4f} ✓")
                else:
                    patience_counter += 1
                    print(f"Epoch {e+1}: Validation Loss = {loss:.4f} "
                          f"(no miglioramento, patience {patience_counter}/{patience})")

                    if patience_counter >= patience:
                        print(f"\nEarly stopping attivato all'epoch {e+1}. "
                              f"Ripristino miglior modello (loss={self.best_loss:.4f}).")
                        self.net = best_model
                        setattr(self.optim, 'net', self.net)
                        break
