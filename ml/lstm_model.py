"""Modelo LSTM para secuencias de sorteos."""
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

WINDOW_SIZE = 30  # sorteos anteriores como input
N_NUMS = 48
DRAW_SIZE = 5


def build_sequences(y: np.ndarray, window: int = WINDOW_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye secuencias deslizantes de longitud `window`.
    Input: y[t-window:t] (shape: window x 48)
    Target: y[t] (shape: 48)
    """
    X_seq, y_seq = [], []
    for i in range(window, len(y)):
        X_seq.append(y[i - window: i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


class LSTMModel:
    def __init__(
        self,
        window: int = WINDOW_SIZE,
        units: int = 64,
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        if not TF_AVAILABLE:
            raise ImportError("tensorflow no instalado")
        self.window = window
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self._train_offset = 0

    def _build(self) -> keras.Model:
        inp = keras.Input(shape=(self.window, N_NUMS))
        x = keras.layers.LSTM(self.units, return_sequences=True)(inp)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.LSTM(self.units // 2)(x)
        x = keras.layers.Dropout(self.dropout)(x)
        out = keras.layers.Dense(N_NUMS, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        # X no se usa — el LSTM trabaja con las secuencias de targets previos
        X_seq, y_seq = build_sequences(y, self.window)
        self._train_offset = self.window
        self.model = self._build()
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
        )
        return self

    def predict_proba(self, X: np.ndarray, y_context: np.ndarray | None = None) -> np.ndarray:
        """
        Si se provee y_context (targets históricos), usa ventana deslizante.
        Si no, retorna probabilidades uniformes como fallback.
        """
        if self.model is None or y_context is None:
            return np.full((X.shape[0], N_NUMS), 1.0 / N_NUMS)

        preds = []
        for i in range(len(X)):
            start = i  # en test set, y_context tiene los targets reales previos
            if start + self.window <= len(y_context):
                window_input = y_context[start: start + self.window]
            else:
                window_input = y_context[-self.window:]
            inp = window_input[np.newaxis, :, :].astype(np.float32)
            pred = self.model.predict(inp, verbose=0)[0]
            preds.append(pred)
        return np.array(preds)
