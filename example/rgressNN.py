
"""
Skeleton - Modello di Regressione
Libreria: libNN (custom)
Task: regressione su dati tuoi
"""

import numpy as np
import matplotlib.pyplot as plt

from libNN.network import NeuralNetwork, Trainer
from libNN.layers import Dense
from libNN.activation import ReLU, Sigmoid, Linear, LeakyReLU
from libNN.losses import MeanSquaredError
from libNN.optimizers import SGDMomentum
from libNN.utils import normalize_data
from libNN.operations import Dropout


# ─────────────────────────────────────────
# 1. IPERPARAMETRI
# ─────────────────────────────────────────

SEED        = 42
EPOCHS      = 200
BATCH_SIZE  = 32
LR          = 0.0001
MOMENTUM    = 0.9
EVAL_EVERY  = 10
TRAIN_RATIO = 0.8   # 80% train, 20% test


# ─────────────────────────────────────────
# 2. DATI
# ─────────────────────────────────────────

def load_data():
    """
    Carica e prepara il dataset.
    TODO: sostituisci con il tuo dataset reale.
    """
    # Placeholder — sostituisci queste righe
    X = np.random.randn(1000, 4)   # (n_samples, n_features)
    y = np.random.randn(1000, 1)   # (n_samples, 1)  <- target continuo
    return X, y


def split_data(X, y, train_ratio=TRAIN_RATIO):
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────
# 3. ARCHITETTURA
# ─────────────────────────────────────────

def build_model() -> NeuralNetwork:
    """
    Definisce l'architettura della rete.
    TODO: modifica neuroni e attivazioni secondo le tue esigenze.
    """
    layers = [
        Dense(neurons=32, activation=ReLU()),
        Dense(neurons=64, activation=LeakyReLU()),
        Dropout(rate=0.2),                        
        Dense(neurons=16, activation=Sigmoid()),
        Dense(neurons=1,  activation=Linear()),   
    ]

    model = NeuralNetwork(
        layers=layers,
        loss=MeanSquaredError(),
        seed=SEED
    )
    return model


# ─────────────────────────────────────────
# 4. TRAINING + VALIDATION (via Trainer)
# ─────────────────────────────────────────

def run_training(model, X_train, y_train, X_test, y_test):
    optimizer = SGDMomentum(lr=LR, momentum=MOMENTUM)
    trainer   = Trainer(model, optimizer)

    trainer.fit(
        X_train, y_train,
        X_test,  y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY
    )
    return trainer


# ─────────────────────────────────────────
# 5. VALUTAZIONE
# ─────────────────────────────────────────

def evaluate(model, X_test, y_test):
    """Metriche sul test set."""
    model.train = False
    y_pred = model.forward(X_test)

    mse    = np.mean((y_test - y_pred) ** 2)
    mae    = np.mean(np.abs(y_test - y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-8)

    print("\n── Risultati sul Test Set ──────────")
    print(f"  MSE : {mse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R2  : {r2:.4f}")
    print("────────────────────────────────────\n")

    return y_pred


# ─────────────────────────────────────────
# 6. PREDIZIONE SU NUOVO DATO
# ─────────────────────────────────────────

def predict_single(model, sample: np.ndarray):
    """
    Predizione su un singolo campione.
    sample shape: (1, n_features)
    """
    model.train = False
    pred = model.forward(sample)
    print(f"Input     : {sample}")
    print(f"Predizione: {pred[0][0]:.4f}")
    return pred


# ─────────────────────────────────────────
# 7. VISUALIZZAZIONE
# ─────────────────────────────────────────

def plot_predictions(y_test, y_pred):
    """Scatter plot: predizioni vs valori reali."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.5, color='coral', label='Predizioni')
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, 'k--', lw=2, label='Ideale')
    plt.title('Predizioni vs Valori Reali')
    plt.xlabel('Target Reale')
    plt.ylabel('Predizione Modello')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Dati
    X, y = load_data()
    # Normalizzazione opzionale — commenta se non serve
    # X = normalize_data(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Modello e training
    model   = build_model()
    trainer = run_training(model, X_train, y_train, X_test, y_test)

    # Valutazione
    y_pred = evaluate(model, X_test, y_test)

    # Predizione su un singolo nuovo dato
    # sostituisci con un esempio reale del tuo problema
    nuovo_dato = np.array([[1.0, -2.0, 0.5, 3.1]])   # shape (1, n_features)
    predict_single(model, nuovo_dato)

    # Plot
    plot_predictions(y_test, y_pred)

