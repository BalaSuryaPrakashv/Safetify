import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()       # collapse spaces
    return text


class Preprocessor:
    def __init__(self, vocab_size: int = 5000, max_len: int = 50):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

    def fit(self, texts):
        cleaned = [clean_text(t) for t in texts]
        self.tokenizer.fit_on_texts(cleaned)

    def transform(self, texts) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")
        return padded

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"Preprocessor loaded from {path}")
        return obj
