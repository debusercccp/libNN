# Guida a libNN

Una libreria per il Deep Learning scritta da zero in Python e NumPy.

---

## Installazione (Consigliata)

Per rendere la libreria importabile da qualsiasi cartella nel tuo sistema senza impazzire con il `PYTHONPATH`, usa il file `pyproject.toml` incluso.

1. Assicurati di avere il file `pyproject.toml` nella cartella radice (`~/libNN/`):
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
    ```
    Bash
    pip install -e .
    ```
Questo ti permette di modificare il codice in libNN/ e vedere i cambiamenti riflettersi immediatamente nei tuoi script senza reinstallare.

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

# 1. Definisci l'architettura come una lista
layers = [
    Dense(neurons=64, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1, activation=Linear())
]

# 2. Inizializza il modello
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
> `Softmax` e `CategoricalCrossEntropy` sono progettate per lavorare **in coppia**: usarle
> separatamente con altre combinazioni può dare risultati inaspettati.

---

## Funzioni di Attivazione

| Attivazione | Quando usarla |
|---|---|
| `ReLU()` | Layer nascosti — scelta predefinita, veloce ed efficace |
| `LeakyReLU(alpha=0.2)` | Layer nascosti — evita il problema dei "neuroni morti" di ReLU |
| `Sigmoid()` | Output per classificazione binaria o segmentazione |
| `Softmax()` | Output per classificazione multi-classe — converte i logit in probabilità (somma = 1) |
| `Linear()` | Output per regressione — lascia passare il valore senza modifiche |

---

## Addestrare il Modello (Il Trainer)

Indipendentemente da quale rete o loss hai scelto, l'addestramento si fa sempre allo stesso modo grazie alla classe `Trainer`.

```python
from libNN.optimizers import SGDMomentum
from libNN.network import Trainer

# Scegli l'ottimizzatore (SGDMomentum è quasi sempre la scelta migliore)
optimizer = SGDMomentum(lr=0.01, momentum=0.9)

# Inizializza il Trainer collegandolo alla rete
trainer = Trainer(net=model, optim=optimizer)

# Lancia l'addestramento
trainer.fit(
    X_train, y_train,   # Dati per imparare
    X_test, y_test,     # Dati per verificare che non stia memorizzando (overfitting)
    epochs=100,         # Quante volte guardare tutti i dati
    eval_every=10,      # Ogni quante epoche stampare la Loss di validazione
    batch_size=32       # Quanti dati processare alla volta
)
```

---

## Esempi per Tipo di Task

### Regressione
```python
from libNN import NeuralNetwork, Dense, ReLU, Linear
from libNN.losses import MeanSquaredError

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1,  activation=Linear()),   # output: valore continuo
]
model = NeuralNetwork(layers=layers, loss=MeanSquaredError())
```

### Classificazione Binaria
```python
from libNN import NeuralNetwork, Dense, ReLU, Sigmoid
from libNN.losses import BinaryCrossEntropy

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1,  activation=Sigmoid()),  # output: probabilità in [0, 1]
]
model = NeuralNetwork(layers=layers, loss=BinaryCrossEntropy())
```

### Classificazione Multi-classe 
```python
from libNN import NeuralNetwork, Dense, ReLU, Softmax
from libNN.losses import CategoricalCrossEntropy

N_CLASSES = 3  # modifica con il tuo numero di classi

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=N_CLASSES, activation=Softmax()),  # output: distribuzione di probabilità
]
model = NeuralNetwork(layers=layers, loss=CategoricalCrossEntropy())

# y deve essere one-hot encoded, es:
# classe 0 → [1, 0, 0]
# classe 1 → [0, 1, 0]
# classe 2 → [0, 0, 1]
```

### Predizione e Valutazione (Multi-classe)
```python
model.train = False
y_probs = model.forward(X_test)          # probabilità per ogni classe
y_pred  = np.argmax(y_probs, axis=1)     # classe con probabilità più alta
```

---

## Utilità (`utils.py`)

```python
from libNN.utils import compute_accuracy, compute_f1_score, mae, rmse, normalize_data
```

| Funzione | Descrizione |
|---|---|
| `compute_accuracy(y_true, y_pred)` | Accuratezza per classificazione |
| `compute_f1_score(y_true, y_pred)` | F1-score per classificazione |
| `mae(y_true, y_pred)` | Mean Absolute Error per regressione |
| `rmse(y_true, y_pred)` | Root Mean Squared Error per regressione |
| `normalize_data(X)` | Normalizza le feature in input |
    epochs=100,         # Quante volte guardare tutti i dati
    eval_every=10,      # Ogni quante epoche stampare la Loss di validazione
    batch_size=32       # Quanti dati processare alla volta
