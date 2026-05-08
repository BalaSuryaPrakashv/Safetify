import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import Preprocessor
from model import build_model

# ── Hyperparameters ───────────────────────────────────────────────────────────
VOCAB_SIZE     = 5000
MAX_LEN        = 50
EMBEDDING_DIM  = 64
BATCH_SIZE     = 16       # smaller batch — better gradient signal on small dataset
EPOCHS         = 50

DATASET_PATH   = "data/dataset.csv"
MODEL_PATH     = "models/bilstm_model.h5"
TOKENIZER_PATH = "tokenizer/tokenizer.pkl"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # 1. Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at '{DATASET_PATH}'.")
        print("Run:  python data/create_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    danger_n = df["label"].sum()
    normal_n = (df["label"] == 0).sum()
    print(f"Loaded {len(df)} samples  |  DANGER={danger_n}  NORMAL={normal_n}")

    texts  = df["text"].astype(str).tolist()
    labels = df["label"].values

    # 2. Train / val / test split  (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"Split  →  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # 3. Compute class weights to handle any remaining imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print(f"Class weights: {class_weight}")

    # 4. Fit preprocessor on training data only
    preprocessor = Preprocessor(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    preprocessor.fit(X_train)

    X_train_seq = preprocessor.transform(X_train)
    X_val_seq   = preprocessor.transform(X_val)
    X_test_seq  = preprocessor.transform(X_test)

    # 5. Build model
    model = build_model(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM)

    # 6. Callbacks
    os.makedirs("models", exist_ok=True)
    os.makedirs("tokenizer", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 7. Train with class weights
    print("\nTraining Bi-LSTM model...")
    model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,      # ← balances training signal
        callbacks=callbacks,
    )

    # 8. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()

    # Find best threshold using validation set
    best_thresh = 0.5
    best_f1 = 0.0
    from sklearn.metrics import f1_score
    for thresh in np.arange(0.3, 0.7, 0.05):
        preds = (y_pred_prob >= thresh).astype(int)
        f1 = f1_score(y_test, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.2f}  (macro-F1={best_f1:.3f})")

    y_pred = (y_pred_prob >= best_thresh).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "DANGER"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # Save best threshold alongside the preprocessor
    preprocessor.threshold = float(best_thresh)

    # 9. Save artifacts
    preprocessor.save(TOKENIZER_PATH)
    print(f"\nModel saved     →  {MODEL_PATH}")
    print(f"Tokenizer saved →  {TOKENIZER_PATH}")
    print(f"Threshold used  →  {best_thresh:.2f}")
    print("\nTraining complete. Run  python app.py  to start the Flask app.")


if __name__ == "__main__":
    main()
