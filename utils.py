import numpy as np
from numpy import ndarray

def assert_same_shape(array: ndarray, array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        f"Shapes must match: {array.shape} vs {array_grad.shape}"

def permute_data(X: ndarray, y: ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def to_one_hot(labels: ndarray, num_classes: int) -> ndarray:
    '''Converte un array di etichette in una matrice one-hot.'''
    labels = labels.astype(int).flatten()
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def compute_accuracy(predictions: ndarray, target: ndarray) -> float:
    '''Calcola la percentuale di risposte corrette (Classificazione).'''
    if predictions.shape != target.shape:
        # Gestisce il caso in cui target siano indici e non one-hot
        pred_labels = np.argmax(predictions, axis=1)
        return np.mean(pred_labels == target.flatten())
    
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(target, axis=1)
    return np.mean(pred_labels == true_labels)

def mae(y_true: ndarray, y_pred: ndarray) -> float:
    '''Mean Absolute Error'''
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray) -> float:
    '''Root Mean Squared Error'''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def normalize_data(data: ndarray) -> ndarray:
    '''Normalizza i dati tra 0 e 1 (molto utile per le immagini).'''
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)

def compute_f1_score(predictions: ndarray, target: ndarray) -> float:
    '''
    Calcola l'F1-Score per la classificazione binaria.
    Si assume che predictions sia già convertito in 0 o 1 (es. preds > 0.5).
    '''
    # Appiattiamo per sicurezza
    preds = predictions.flatten()
    actual = target.flatten()

    # Calcolo True Positives, False Positives, False Negatives
    tp = np.sum((preds == 1) & (actual == 1))
    fp = np.sum((preds == 1) & (actual == 0))
    fn = np.sum((preds == 0) & (actual == 1))

    # Calcolo Precision e Recall con un piccolo epsilon per evitare divisioni per zero
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    # F1 Score = 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return f1
