import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout
)


def build_model(vocab_size: int, max_len: int, embedding_dim: int = 64) -> tf.keras.Model:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),

        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),

        Bidirectional(LSTM(32)),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.summary()
    return model
