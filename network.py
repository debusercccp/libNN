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
    '''
    A sequential neural network class.
    Assembles layers and provides forward/backward propagation.
    '''
    def __init__(self, layers: List[Layer], loss: Loss, seed: int = None):
        '''
        Initialize NeuralNetwork.
        
        Args:
            layers: List of Layer objects, applied sequentially.
            loss: Loss function (MeanSquaredError, BinaryCrossEntropy, etc.)
            seed: Random seed for weight initialization. If provided, sets seed for all layers.
        '''
        self.layers = layers
        self.loss = loss
        self.seed = seed
        self.train = True  # Flag to control Dropout and other training-specific behavior
        self.best_loss = float('inf')  # Tracks best validation loss for checkpointing
        
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)        

    def forward(self, x_batch: ndarray) -> ndarray:
        '''
        Forward pass through all layers.
        
        Args:
            x_batch: Input batch, shape (batch_size, input_dim)
            
        Returns:
            Output batch after passing through all layers.
        '''
        x_out = x_batch
        for layer in self.layers:
            # Pass training flag to layers that support it (e.g., Dropout)
            if hasattr(layer, 'is_training'):
                layer.is_training = self.train
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
        Backward pass through all layers.
        Computes gradients for all parameters via chain rule.
        
        Args:
            loss_grad: Gradient of loss w.r.t. network output (from loss.backward())
        '''
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        '''
        Single training step: forward, compute loss, backward.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Loss value for this batch.
        '''
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    # --- MODEL SAVING/LOADING ---

    def save_model(self, filename="model.pkl", is_best=False):
        '''
        Save entire network structure and parameters using pickle.
        Preserves layers, loss function, seed, and best_loss tracking.
        
        Args:
            filename: Path to save to (default "model.pkl")
            is_best: If True, prints special message indicating this is the best model so far.
        '''
        with open(filename, 'wb') as f:
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
        '''
        Load a saved model from pickle file and reconstruct NeuralNetwork instance.
        
        Args:
            filename: Path to pickled model file.
            
        Returns:
            Reconstructed NeuralNetwork instance ready for inference or training.
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
        '''
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} non trovato.")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        nn = NeuralNetwork(layers=data['layers'], loss=data['loss'], seed=data.get('seed'))
        nn.best_loss = data.get('best_loss', float('inf'))
        
        print(f" Modello caricato con successo da {filename}.")
        return nn

    def check_and_save(self, current_loss: float, filename="best_model.pkl"):
        '''
        Utility for auto-checkpoint: saves model only if current_loss improved best_loss.
        Useful to call during training to keep the best checkpoint.
        
        Args:
            current_loss: Current validation loss to check against best.
            filename: Path to save best model to.
        '''
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_model(filename, is_best=True)

    # --- PARAMETER ACCESS (for optimizers) ---

    def params(self):
        '''
        Generator yielding all learnable parameters in the network.
        Used by optimizers to update weights and biases.
        '''
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params is not None:
                yield from layer.params

    def param_grads(self):
        '''
        Generator yielding gradients of all learnable parameters.
        Used by optimizers to compute parameter updates.
        '''
        for layer in self.layers:
            if hasattr(layer, 'param_grads') and layer.param_grads is not None:
                yield from layer.param_grads

# --- Trainer Class ---

class Trainer(object):
    '''
    Trainer class encapsulates the training loop.
    Handles batch generation, loss evaluation, early stopping, and checkpoint management.
    '''
    def __init__(self, net: NeuralNetwork, optim) -> None:
        '''
        Initialize Trainer.
        
        Args:
            net: NeuralNetwork instance to train.
            optim: Optimizer instance (SGD, SGDMomentum, etc.)
        '''
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        # Attach network to optimizer so it can access params()
        setattr(self.optim, 'net', self.net)
        
    def generate_batches(self, X: ndarray, y: ndarray, size: int = 32):
        '''
        Generator yielding batches of data.
        
        Args:
            X: Features array, shape (n_samples, n_features)
            y: Target array, shape (n_samples, ...)
            size: Batch size (default 32)
            
        Yields:
            Tuples of (X_batch, y_batch)
        '''
        assert X.shape[0] == y.shape[0], \
        f"Features ha {X.shape[0]} samples, target ne ha {y.shape[0]}"
        
        N = X.shape[0]
        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]
            yield X_batch, y_batch

    def _evaluate(self, X_test: ndarray, y_test: ndarray) -> float:
        '''
        Evaluate model on test data (with Dropout disabled).
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Loss value on test set.
        '''
        self.net.train = False  # Disable Dropout
        test_preds = self.net.forward(X_test)
        loss_val = self.net.loss.forward(test_preds, y_test)
        self.net.train = True  # Re-enable training mode
        return loss_val
            
    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int = 100, eval_every: int = 10,
            batch_size: int = 32, seed: int = 1,
            restart: bool = True, patience: int = 5) -> None:
        '''
        Train the neural network with early stopping and best-model checkpointing.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Validation features (used for early stopping)
            y_test: Validation targets
            epochs: Maximum number of training epochs
            eval_every: Evaluate on validation set every N epochs
            batch_size: Mini-batch size for SGD
            seed: Random seed for reproducible shuffling
            restart: If True, reinitialize layer parameters before training
            patience: Number of eval steps without improvement before early stopping
            
        Training stops early if validation loss doesn't improve for `patience` evaluations.
        Best model is restored at the end.
        
        Example:
            >>> trainer.fit(X_train, y_train, X_test, y_test,
            ...             epochs=100, eval_every=10, batch_size=32, patience=5)
        '''

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        best_model = None
        patience_counter = 0

        for e in range(epochs):
            # Shuffle training data each epoch
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            # Train on all batches
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            # Periodic validation
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

                    # Early stopping trigger
                    if patience_counter >= patience:
                        print(f"\nEarly stopping attivato all'epoch {e+1}. "
                              f"Ripristino miglior modello (loss={self.best_loss:.4f}).")
                        self.net = best_model
                        setattr(self.optim, 'net', self.net)
                        break
