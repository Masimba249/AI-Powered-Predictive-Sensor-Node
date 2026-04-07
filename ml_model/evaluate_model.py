"""
evaluate_model.py
─────────────────
Benchmarks the TFLite model on the test set and computes:
  - Accuracy, Precision, Recall, F1
  - Inference latency (simulated)
  - Model size vs accuracy tradeoff
  - Per-class ROC curves

Usage:
    python evaluate_model.py --model model_int8.tflite
"""

import argparse
import time
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from train_model import load_and_window_data, CLASSES
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="model_int8.tflite")
    p.add_argument("--scaler", default="scaler.pkl")
    return p.parse_args()

def run_tflite_inference(interpreter, X):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    predictions    = []

    for i in range(len(X)):
        inp = X[i:i+1]

        # Handle int8 quantisation
        if input_details[0]["dtype"] == np.int8:
            scale, zp = input_details[0]["quantization"]
            inp = (inp / scale + zp).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.int8:
            scale, zp = output_details[0]["quantization"]
            out = (out.astype(np.float32) - zp) * scale

        predictions.append(out[0])

    return np.array(predictions)

def main():
    args = parse_args()

    print("[1/4] Loading data...")
    X, y_str = load_and_window_data()
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(y_str)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)

    print(f"[2/4] Loading TFLite model: {args.model}")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    print("[3/4] Running inference...")
    start = time.perf_counter()
    probs = run_tflite_inference(interpreter, X_test_scaled)
    elapsed = time.perf_counter() - start

    y_pred = np.argmax(probs, axis=1)

    print("\n[4/4] Results:")
    print(f"  Samples      : {len(X_test)}")
    print(f"  Total time   : {elapsed*1000:.1f} ms")
    print(f"  Per-sample   : {elapsed/len(X_test)*1000:.3f} ms")
    print(f"  Model size   : {open(args.model,'rb').read().__len__()/1024:.1f} KB")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # ROC curves (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(len(CLASSES))))
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves — Predictive Maintenance Classifier")
    plt.legend(); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("[PLOT] ROC curves saved → roc_curves.png")

if __name__ == "__main__":
    main()