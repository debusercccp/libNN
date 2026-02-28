"""
Modello di Classificazione Multi-classe
Libreria: libNN (custom)
Output: Softmax + CrossEntropy
"""

import numpy as np
import matplotlib.pyplot as plt

from libNN.network import NeuralNetwork, Trainer
from libNN.layers import Dense
from libNN.activation import ReLU, Sigmoid, LeakyReLU, Softmax  
from libNN.losses import CategoricalCrossEntropy          
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
TRAIN_RATIO = 0.8

N_CLASSES   = 3   


# ─────────────────────────────────────────
# 2. DATI
# ─────────────────────────────────────────

def load_data(n_classes=N_CLASSES):
    """
    Carica e prepara il dataset.
    TODO: sostituisci con il tuo dataset reale.

    y deve essere one-hot encoded → shape (n_samples, n_classes)
    Es: classe 2 su 3 → [0, 0, 1]
    """
    # Placeholder — sostituisci queste righe
    n_samples  = 1000
    n_features = 4

    X = np.random.randn(n_samples, n_features)

    # Classi come interi (0, 1, 2, ...)
    y_int = np.random.randint(0, n_classes, size=n_samples)

    # One-hot encoding
    y = one_hot(y_int, n_classes)

    return X, y, y_int


def one_hot(y_int: np.ndarray, n_classes: int) -> np.ndarray:
    """Converte etichette intere in one-hot. Es: 2 → [0, 0, 1]"""
    ohe = np.zeros((len(y_int), n_classes))
    ohe[np.arange(len(y_int)), y_int] = 1
    return ohe


def split_data(X, y, y_int, train_ratio=TRAIN_RATIO):
    split = int(len(X) * train_ratio)
    return (
        X[:split],    X[split:],
        y[:split],    y[split:],
        y_int[:split], y_int[split:]
    )


# ─────────────────────────────────────────
# 3. ARCHITETTURA
# ─────────────────────────────────────────

def build_model(n_classes=N_CLASSES) -> NeuralNetwork:
    """
    Rete per classificazione multi-classe.
    Ultimo layer: Dense(neurons=n_classes, activation=Softmax())
    TODO: modifica neuroni e attivazioni secondo le tue esigenze.
    """
    layers = [
        Dense(neurons=32, activation=ReLU()),
        Dense(neurons=64, activation=LeakyReLU()),
        Dropout(rate=0.2),                                      
        Dense(neurons=32, activation=ReLU()),
        Dense(neurons=n_classes, activation=Softmax()), 
    ]

    model = NeuralNetwork(
        layers=layers,
        loss=CategoricalCrossEntropy(),          
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

def evaluate(model, X_test, y_test_ohe, y_test_int):
    """Accuracy e matrice di confusione sul test set."""
    model.train = False
    y_probs = model.forward(X_test)               # output softmax: (n_samples, n_classes)
    y_pred  = np.argmax(y_probs, axis=1)          # classe predetta

    accuracy = np.mean(y_pred == y_test_int)

    print("\n── Risultati sul Test Set ──────────")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print("────────────────────────────────────\n")

    return y_pred, y_probs


# ─────────────────────────────────────────
# 6. PREDIZIONE SU NUOVO DATO
# ─────────────────────────────────────────

def predict_single(model, sample: np.ndarray, class_names: list = None):
    """
    Predizione su un singolo campione.
    sample shape: (1, n_features)
    """
    model.train = False
    probs       = model.forward(sample)
    classe      = np.argmax(probs, axis=1)[0]
    label       = class_names[classe] if class_names else classe

    print(f"Input          : {sample}")
    print(f"Classe predetta: {label}")
    print(f"Probabilità    : { {i: f'{p:.4f}' for i, p in enumerate(probs[0])} }")
    return classe, probs


# ─────────────────────────────────────────
# 7. VISUALIZZAZIONE
# ─────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, n_classes=N_CLASSES, class_names=None):
    """Matrice di confusione manuale (senza sklearn)."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    labels = class_names if class_names else [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predetto')
    ax.set_ylabel('Reale')
    ax.set_title('Matrice di Confusione')

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    plt.show()


def plot_class_probabilities(y_probs, y_true, n_samples=100):
    """Mostra le probabilità predette per i primi n_samples del test set."""
    plt.figure(figsize=(10, 4))
    for c in range(y_probs.shape[1]):
        plt.plot(y_probs[:n_samples, c], alpha=0.7, label=f'Classe {c}')
    plt.title(f'Probabilità Softmax (primi {n_samples} campioni)')
    plt.xlabel('Campione')
    plt.ylabel('Probabilità')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Nomi delle classi (opzionale — puoi lasciare None)
    # sostituisci con i tuoi nomi reali
    class_names = ["Classe A", "Classe B", "Classe C"]   # deve avere N_CLASSES elementi

    # Dati
    X, y_ohe, y_int = load_data()
    # Normalizzazione opzionale — decommentala se serve
    # X = normalize_data(X)

    X_train, X_test, y_train, y_test, y_train_int, y_test_int = split_data(X, y_ohe, y_int)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Modello e training
    model   = build_model()
    trainer = run_training(model, X_train, y_train, X_test, y_test)

    # Valutazione
    y_pred, y_probs = evaluate(model, X_test, y_test, y_test_int)

    # Predizione su un singolo nuovo dato
    # sostituisci con un esempio reale del tuo problema
    nuovo_dato = np.array([[1.0, -2.0, 0.5, 3.1]])   # shape (1, n_features)
    predict_single(model, nuovo_dato, class_names)

    # Plot
    plot_confusion_matrix(y_test_int, y_pred, N_CLASSES, class_names)
    plot_class_probabilities(y_probs, y_test_int)
