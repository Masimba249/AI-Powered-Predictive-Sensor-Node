"""
train_model.py
──────────────
Full ML pipeline:
  1. Load + merge CSV datasets
  2. Feature engineering (statistical features per window)
  3. Train a compact MLP classifier
  4. Evaluate + confusion matrix
  5. Export to TFLite (float32 + int8 quantized)
  6. Generate C header (model_data.h) for STM32 firmware

Requirements:
    pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pickle

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR       = "data_collection/data"
OUTPUT_DIR     = "."
MODEL_PATH     = os.path.join(OUTPUT_DIR, "model.tflite")
QUANT_PATH     = os.path.join(OUTPUT_DIR, "model_int8.tflite")
HEADER_PATH    = "../firmware/model_data.h"
SCALER_PATH    = os.path.join(OUTPUT_DIR, "scaler.pkl")

WINDOW_SIZE    = 100   # Samples per window (1s @ 100Hz)
WINDOW_STEP    = 50    # 50% overlap
FEATURE_SIZE   = 74
CLASSES        = ["normal", "bearing_fault", "imbalance", "looseness"]
NUM_CLASSES    = len(CLASSES)

# ── Feature extraction (mirrors firmware/ml_inference.c) ─────────────

def extract_features_python(window: np.ndarray) -> np.ndarray:
    """
    window: shape (WINDOW_SIZE, 7) → [ax, ay, az, gx, gy, gz, temp]
    Returns: feature vector of shape (FEATURE_SIZE,)
    """
    axes = window[:, :6]   # ax,ay,az,gx,gy,gz
    temp = window[:, 6]

    feats = []

    for a in range(6):
        col  = axes[:, a]
        m    = col.mean()
        rms  = np.sqrt((col**2).mean())
        std  = col.std()
        pk   = np.abs(col).max()
        ptp  = col.max() - col.min()
        kurt = ((((col - m) / (std + 1e-9))**4).mean()) - 3.0
        skew = (((col - m) / (std + 1e-9))**3).mean()
        crest = pk / (rms + 1e-9)
        feats.extend([m, rms, std, pk, kurt, skew, crest, ptp])

    # Magnitude RMS
    feats.append(np.sqrt((axes[:, :3]**2).sum(axis=1).mean()))   # accel
    feats.append(np.sqrt((axes[:, 3:]**2).sum(axis=1).mean()))   # gyro

    # Temperature stats
    feats.append(temp.mean())
    feats.append(temp.max())
    feats.append(temp.max() - temp[0])

    # Zero-crossing rate per axis
    for a in range(6):
        col = axes[:, a]
        zc  = np.sum(np.diff(np.sign(col)) != 0) / WINDOW_SIZE
        feats.append(zc)

    # SMA
    feats.append(np.abs(axes[:, :3]).sum(axis=1).mean())

    # Cross-axis correlation
    feats.append(np.cov(axes[:,0], axes[:,1])[0,1])
    feats.append(np.cov(axes[:,0], axes[:,2])[0,1])
    feats.append(np.cov(axes[:,1], axes[:,2])[0,1])

    v = np.array(feats, dtype=np.float32)
    # Pad or truncate to FEATURE_SIZE
    if len(v) < FEATURE_SIZE:
        v = np.pad(v, (0, FEATURE_SIZE - len(v)))
    return v[:FEATURE_SIZE]

# ── Data loading ──────────────────────────────────────────────────────

def load_and_window_data():
    all_features, all_labels = [], []

    for class_name in CLASSES:
        csvfiles = glob.glob(os.path.join(DATA_DIR, f"{class_name}_*.csv"))
        print(f"  [{class_name}] {len(csvfiles)} file(s) found")

        for f in csvfiles:
            df = pd.read_csv(f)
            # Columns: timestamp_ms, ax, ay, az, gx, gy, gz, temp, label
            data = df[["ax","ay","az","gx","gy","gz","temp"]].values.astype(np.float32)

            # Sliding window
            for start in range(0, len(data) - WINDOW_SIZE, WINDOW_STEP):
                w = data[start:start + WINDOW_SIZE]
                all_features.append(extract_features_python(w))
                all_labels.append(class_name)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    print(f"\n[DATA] Total windows: {len(X)} | Features: {X.shape[1]}")
    return X, y

# ── Model architecture ────────────────────────────────────────────────

def build_model(input_size: int, num_classes: int) -> keras.Model:
    """
    Compact MLP optimised for Cortex-M4 (< 50KB when quantised).
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_size,)),
        keras.layers.Dense(128, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ], name="predictive_maintenance_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model

# ── TFLite export ─────────────────────────────────────────────────────

def export_tflite_float(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] Float32 TFLite: {path} ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model

def export_tflite_int8(model, X_train, path):
    def representative_dataset():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] INT8 quantised TFLite: {path} ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model

def generate_c_header(tflite_bytes, header_path, array_name="g_model"):
    hex_array = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    c_code = f"""/**
 * @file  model_data.h
 * @brief Auto-generated TFLite Micro model data.
 *        Generated by train_model.py
 *        DO NOT EDIT MANUALLY
 */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

/* Model stored in flash */
const uint8_t {array_name}[] __attribute__((aligned(8))) = {{
  {hex_array}
}};

const unsigned int {array_name}_len = {len(tflite_bytes)};

#endif /* MODEL_DATA_H */
"""
    os.makedirs(os.path.dirname(os.path.abspath(header_path)), exist_ok=True)
    with open(header_path, "w") as f:
        f.write(c_code)
    print(f"[EXPORT] C header written to {header_path}")

# ── Main pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Predictive Maintenance — TinyML Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading and windowing data...")
    X, y_str = load_and_window_data()

    # 2. Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(y_str)
    print(f"[2/6] Label distribution: { {c: int((y==i).sum()) for i,c in enumerate(CLASSES)} }")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[3/6] Scaler saved → {SCALER_PATH}")

    # 5. Train
    print("\n[4/6] Training model...")
    model = build_model(FEATURE_SIZE, NUM_CLASSES)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluate
    print("\n[5/6] Evaluating...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.title("Confusion Matrix — Predictive Maintenance Model")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[PLOT] Confusion matrix saved → confusion_matrix.png")

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"],     label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"],     label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)

    # 7. Export
    print("\n[6/6] Exporting TFLite models...")
    float_bytes = export_tflite_float(model, MODEL_PATH)
    int8_bytes  = export_tflite_int8(model, X_train[:200], QUANT_PATH)
    generate_c_header(int8_bytes, HEADER_PATH)

    print("\n✅ Pipeline complete!")
    print(f"   Float32 model : {MODEL_PATH}")
    print(f"   INT8 model    : {QUANT_PATH}")
    print(f"   C header      : {HEADER_PATH}")
    print(f"   Scaler        : {SCALER_PATH}")

if __name__ == "__main__":
    main()