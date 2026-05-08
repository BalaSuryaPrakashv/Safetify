import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import Preprocessor, clean_text

MODEL_PATH     = "models/bilstm_model.h5"
TOKENIZER_PATH = "tokenizer/tokenizer.pkl"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'.")
        print("        Run:  python src/train.py")
        sys.exit(1)
    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ERROR] Tokenizer not found at '{TOKENIZER_PATH}'.")
        print("        Run:  python src/train.py")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = Preprocessor.load(TOKENIZER_PATH)
    return model, preprocessor


def predict(text: str, model, preprocessor) -> dict:
    threshold = getattr(preprocessor, "threshold", 0.5)
    sequence  = preprocessor.transform([text])
    prob      = float(model.predict(sequence, verbose=0)[0][0])

    if prob >= threshold:
        label      = "DANGER"
        confidence = prob
    else:
        label      = "NORMAL"
        confidence = 1.0 - prob

    return {"label": label, "confidence": round(confidence * 100, 2), "raw_score": round(prob, 4)}


def main():
    print("=" * 50)
    print("  Women Safety Threat Detection System")
    print("  Powered by Bi-LSTM")
    print("=" * 50)
    print("Loading model...")

    model, preprocessor = load_artifacts()

    print("Model ready.\n")
    print("Type a phrase and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye. Stay safe.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye. Stay safe.")
            break

        result = predict(user_input, model, preprocessor)

        if result["label"] == "DANGER":
            print(f"  ⚠  [DANGER]  Confidence: {result['confidence']}%\n")
        else:
            print(f"  ✓  [NORMAL]  Confidence: {result['confidence']}%\n")


if __name__ == "__main__":
    main()
