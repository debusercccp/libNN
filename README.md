# Guida a libNN

Una libreria per il Deep Learning scritta da zero in Python e NumPy.

---

## Versione 0.1.0 - Correzioni e Miglioramenti

**Novità in questa release:**
- ✅ **Dropout fix**: Mask ora sempre inizializzato, garantisce correttezza in eval mode
- ✅ **He Initialization**: Dense layer usa `sqrt(2.0 / in_dim)` ottimale per ReLU
- ✅ **Seed handling**: Seed 0 ora rilevato correttamente (fix bug `is not None`)
- ✅ **Exported utility**: `to_one_hot()` aggiunto a `__init__.py`
- ✅ **Docstring completi**: Tutti i moduli documentati con esempi e note

---

## Installazione (Consigliata)

Per rendere la libreria importabile da qualsiasi cartella nel tuo sistema senza impazzire con il `PYTHONPATH`, usa il file `pyproject.toml` incluso.

Assicurati di avere il file `pyproject.toml` nella cartella radice (`~/libNN/`):

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "libNN"
version = "0.1.0"
dependencies = ["numpy", "matplotlib"]

[tool.setuptools]
packages = ["libNN"]
package-dir = {"libNN" = "."}
```

Dalla radice della libreria, installa in modalità editable:

```bash
pip install -e .
```

Questo ti permette di modificare il codice in `libNN/` e vedere i cambiamenti riflettersi immediatamente nei tuoi script senza reinstallare.

---

## Struttura della Libreria

- **`network.py`**: Contiene `NeuralNetwork` (per assemblare i layer) e `Trainer` (per il ciclo di addestramento).
- **`losses.py`**: Contiene le funzioni di Loss (`MeanSquaredError`, `BinaryCrossEntropy`, `DiceLoss`, `CategoricalCrossEntropy`).
- **`layers.py`**: I layer standard, come `Dense` (i classici neuroni completamente connessi).
- **`activation.py`**: Le funzioni di attivazione (`ReLU`, `LeakyReLU`, `Sigmoid`, `Linear`, `Softmax`) che danno "intelligenza" e non-linearità alla rete.
- **`optimizers.py`**: I motori di ricerca del minimo (`SGD`, `SGDMomentum`) che aggiornano i pesi.
- **`operations.py`**: Operazioni speciali come il `Dropout` (per la regolarizzazione).

---

## Costruire una Rete Standard (Feedforward)

Usa la classe `NeuralNetwork` combinata con i layer `Dense` quando hai dati tabulari (es. file CSV) o problemi matematici classici (come la regressione).

```python
from libNN import NeuralNetwork, Dense, ReLU, Dropout, Linear
from libNN.losses import MeanSquaredError

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1, activation=Linear())
]

model = NeuralNetwork(layers=layers, loss=MeanSquaredError())
```

---

## Scegliere la Loss Giusta

La funzione di Loss è l'obiettivo della tua rete. Se scegli quella sbagliata, la rete non imparerà nulla.

| Loss | Usala per | Ultimo layer |
|---|---|---|
| `MeanSquaredError()` | **Regressione** — predire un numero continuo (es. temperatura, prezzo di una casa) | `Linear()` |
| `BinaryCrossEntropy()` | **Classificazione Binaria** — due classi (es. cane/gatto, sfondo/primo piano) | `Sigmoid()` |
| `DiceLoss()` | **Segmentazione Sbilanciata** — trovare piccole regioni in immagini grandi | `Sigmoid()` |
| `CategoricalCrossEntropy()` | **Classificazione Multi-classe** — tre o più classi mutualmente esclusive | `Softmax()` |

> `CategoricalCrossEntropy` richiede che le etichette `y` siano in formato **one-hot encoded**
> (es. classe 2 su 3 classi → `[0, 0, 1]`).

---

## ⚠️ IMPORTANTE: Softmax + CategoricalCrossEntropy

**Softmax** e **CategoricalCrossEntropy** sono **progettati per lavorare in coppia**. Quando usati insieme, il gradiente della combinazione semplifica a `(prediction - target)`, che è numericamente stabile e altamente efficiente.

```python
# ✅ CORRETTO: Usa sempre insieme
layers = [
    Dense(64, ReLU()),
    Dense(n_classes, Softmax())  # Output: probabilità
]
model = NeuralNetwork(layers, loss=CategoricalCrossEntropy())

# ❌ NON FARLO: Usarli separatamente con altre combinazioni
# layers = [..., Dense(n_classes, Sigmoid())]  # ← sbagliato activation
# model = NeuralNetwork(layers, loss=CategoricalCrossEntropy())  # ← incompatibile
```

Se usi questa coppia, il training sarà **più veloce e stabile** perché la backprop è ottimizzata.

---

## Funzioni di Attivazione

| Attivazione | Quando usarla |
|---|---|
| `ReLU()` | Layer nascosti — scelta predefinita, veloce ed efficace |
| `LeakyReLU(alpha=0.2)` | Layer nascosti — evita il problema dei "neuroni morti" di ReLU |
| `Sigmoid()` | Output per classificazione binaria o segmentazione |
| `Softmax()` | Output per classificazione multi-classe — converte i logit in probabilità (somma = 1). **Usa sempre con CategoricalCrossEntropy** |
| `Linear()` | Output per regressione — lascia passare il valore senza modifiche |

### He vs Glorot Initialization (Dettagli Tecnici)

La libreria usa **He initialization** nei layer Dense (`sqrt(2.0 / in_dim)`), che è ottimale per ReLU e varianti:

```python
scale = np.sqrt(2.0 / in_dim)  # He initialization
self.params.append(np.random.standard_normal((in_dim, out_dim)) * scale)
```

Se preferisci **Glorot initialization** (per Sigmoid/Tanh), puoi creare un custom layer:

```python
scale = np.sqrt(1.0 / (in_dim + out_dim))  # Glorot
```

Per questa libreria, **mantieni He con ReLU** — è la scelta migliore.

---

## Addestrare il Modello (Il Trainer)

Indipendentemente da quale rete o loss hai scelto, l'addestramento si fa sempre allo stesso modo grazie alla classe `Trainer`.

```python
from libNN.optimizers import SGDMomentum
from libNN.network import Trainer

optimizer = SGDMomentum(lr=0.01, momentum=0.9)
trainer = Trainer(net=model, optim=optimizer)

trainer.fit(
    X_train, y_train,   # Dati per imparare
    X_test,  y_test,    # Dati per verificare che non stia memorizzando (overfitting)
    epochs=100,         # Quante volte guardare tutti i dati
    eval_every=10,      # Ogni quante epoche valutare la Loss di validazione
    batch_size=32,      # Quanti dati processare alla volta
    patience=5          # Epoche di valutazione senza miglioramento prima di fermarsi
)
```

### Parametri di `fit()`

| Parametro | Default | Descrizione |
|---|---|---|
| `epochs` | `100` | Numero massimo di epoche di addestramento |
| `eval_every` | `10` | Ogni quante epoche valutare sul validation set |
| `batch_size` | `32` | Dimensione dei mini-batch |
| `seed` | `1` | Seed per la riproducibilità dello shuffle |
| `restart` | `True` | Se `True`, reinizializza i layer prima di partire |
| `patience` | `5` | Numero di valutazioni consecutive senza miglioramento prima dell'early stopping |

### Early Stopping

Il `Trainer` monitora la validation loss ad ogni valutazione. Se la loss non migliora per `patience` valutazioni consecutive, l'addestramento si interrompe automaticamente e il modello viene riportato al miglior checkpoint registrato.

```
Epoch 10: Validation Loss = 0.0842 ✓
Epoch 20: Validation Loss = 0.0761 ✓
Epoch 30: Validation Loss = 0.0798 (no miglioramento, patience 1/5)
Epoch 40: Validation Loss = 0.0823 (no miglioramento, patience 2/5)
...
Early stopping attivato all'epoch 80. Ripristino miglior modello (loss=0.0761).
```

> Con `eval_every=10` e `patience=5`, il training si ferma al massimo dopo 50 epoche consecutive senza miglioramento.

---

## Salvataggio e Caricamento

Il salvataggio preserva l'intera struttura dei layer, i pesi, i bias e lo stato dell'addestramento tramite `pickle`.

```python
# Salva il modello corrente
model.save_model("mio_modello.pkl")

# Carica un modello esistente
model = NeuralNetwork.load_model("mio_modello.pkl")
```

### Auto-Checkpoint (Best Model)

`check_and_save()` salva il modello **solo se** la loss corrente è inferiore al minimo storico. Utile da chiamare manualmente nel proprio loop di training.

```python
# Salva solo se è il miglior modello finora
model.check_and_save(current_loss=val_loss, filename="best_model.pkl")
```

---

## Dropout: Regolarizzazione Contro l'Overfitting

`Dropout` disattiva casualmente neuroni durante il training per ridurre il co-adattamento.

```python
from libNN import Dropout

layers = [
    Dense(64, ReLU()),
    Dropout(keep_prob=0.8),  # Disattiva ~20% dei neuroni
    Dense(32, ReLU()),
    Dropout(keep_prob=0.8),
    Dense(n_classes, Softmax())
]
```

**Nota**: Dropout è automaticamente disattivato durante la valutazione (`model.train = False`). Non devi fare nulla di speciale — la libreria lo gestisce.

---

## Utilità (`utils.py`)

```python
from libNN import compute_accuracy, compute_f1_score, mae, rmse, normalize_data, to_one_hot
```

| Funzione | Descrizione |
|---|---|
| `compute_accuracy(y_true, y_pred)` | Accuratezza per classificazione |
| `compute_f1_score(y_true, y_pred)` | F1-score per classificazione binaria |
| `mae(y_true, y_pred)` | Mean Absolute Error per regressione |
| `rmse(y_true, y_pred)` | Root Mean Squared Error per regressione |
| `normalize_data(X)` | Normalizza le feature in input a [0, 1] |
| `to_one_hot(labels, n_classes)` | Converte etichette in one-hot encoding |

---

## Esempi per Tipo di Task

### Regressione

```python
from libNN import NeuralNetwork, Dense, ReLU, Linear, to_one_hot
from libNN.losses import MeanSquaredError
from libNN.optimizers import SGDMomentum
from libNN.network import Trainer

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1,  activation=Linear()),
]
model = NeuralNetwork(layers=layers, loss=MeanSquaredError())

trainer = Trainer(net=model, optim=SGDMomentum(lr=0.01, momentum=0.9))
trainer.fit(X_train, y_train, X_test, y_test, epochs=100, patience=10)
```

### Classificazione Binaria

```python
from libNN import NeuralNetwork, Dense, ReLU, Sigmoid, Dropout
from libNN.losses import BinaryCrossEntropy
from libNN.optimizers import SGDMomentum
from libNN.network import Trainer

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=32, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=1,  activation=Sigmoid()),
]
model = NeuralNetwork(layers=layers, loss=BinaryCrossEntropy())

trainer = Trainer(net=model, optim=SGDMomentum(lr=0.01, momentum=0.9))
trainer.fit(X_train, y_train, X_test, y_test, epochs=100, patience=10)
```

### Classificazione Multi-classe

```python
from libNN import (NeuralNetwork, Dense, ReLU, Softmax, Dropout, 
                   to_one_hot, compute_accuracy)
from libNN.losses import CategoricalCrossEntropy
from libNN.optimizers import SGDMomentum
from libNN.network import Trainer
import numpy as np

N_CLASSES = 3

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=32, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=N_CLASSES, activation=Softmax()),
]
model = NeuralNetwork(layers=layers, loss=CategoricalCrossEntropy())

# Converti etichette in one-hot
y_train_onehot = to_one_hot(y_train, N_CLASSES)
y_test_onehot = to_one_hot(y_test, N_CLASSES)

trainer = Trainer(net=model, optim=SGDMomentum(lr=0.01, momentum=0.9))
trainer.fit(X_train, y_train_onehot, X_test, y_test_onehot, 
            epochs=100, patience=10)
```

### Predizione (Multi-classe)

```python
model.train = False  # Disabilita Dropout
y_probs = model.forward(X_test)          # probabilità per ogni classe
y_pred  = np.argmax(y_probs, axis=1)     # classe con probabilità più alta

# Valuta
accuracy = compute_accuracy(y_probs, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Troubleshooting

### **Problema: La loss non diminuisce**

**Soluzioni (in ordine):**
1. Aumenta il learning rate (`lr=0.1`, `lr=0.05`)
2. Riduci il batch size (`batch_size=16` invece di 32)
3. Cambia optimizer (prova `SGDMomentum` invece di `SGD`)
4. Normalizza i dati: `X_train = normalize_data(X_train)`
5. Aumenta il numero di epoche o riduci patience

### **Problema: Overfitting (train loss bassa, validation loss alta)**

**Soluzioni:**
1. Aumenta il Dropout: `Dropout(keep_prob=0.5)` invece di 0.8
2. Riduci la complessità del modello (meno neuroni, meno layer)
3. Aumenta la quantità di dati di training
4. Usa early stopping (che la libreria fa automaticamente)

### **Problema: Errore `AttributeError: 'Dropout' has no attribute 'mask'`**

**Risolto nella versione 0.1.0**: La mask è ora sempre inizializzata, anche in eval mode.

### **Problema: Seed 0 non funziona**

**Risolto nella versione 0.1.0**: Ora usa `is not None` invece di truthiness check.

---

## Note di Sviluppo

- **Autograd**: Implementazione manuale di backprop (no `autograd`, no `JAX`) — ogni operazione ha `_output()` e `_input_grad()` scritti esplicitamente.
- **Performance**: Per dataset molto grandi (>1M campioni), considera `PyTorch` o `TensorFlow`.
- **Stabilità numerica**: Tutti i logaritmi usano clipping per evitare `log(0)`.

---

## Licenza e Attribuzione

libNN © 2024 - Noya (Rocco)

Sviluppata per scopi educativi e di ricerca. Senti libero di usarla, modificarla, e condividere miglioramenti!
