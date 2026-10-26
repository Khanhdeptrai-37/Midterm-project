import os
import sys
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ml_dir = r"d:\download\model\ML"
mlp_dir = r"d:\download\model\MLP"
sys.path.append(mlp_dir)
from model import ModulationMLP

print("=== Loading ML & MLP models ===")
t0 = time.time()
svm = joblib.load(os.path.join(ml_dir, "svm_model.pkl"))
print(f"SVM loaded in {time.time() - t0:.2f}s")

t0 = time.time()
rf = joblib.load(os.path.join(ml_dir, "rf_model.pkl"))
print(f"RF loaded in {time.time() - t0:.2f}s")

scaler = joblib.load(os.path.join(ml_dir, "scaler.pkl"))
le = joblib.load(os.path.join(ml_dir, "label_encoder.pkl"))

with open(os.path.join(mlp_dir, "class_mapping.pkl"), "rb") as f:
    mapping = pickle.load(f)

mlp = ModulationMLP(18, 8)
mlp.load_state_dict(torch.load(os.path.join(mlp_dir, "best_mlp.pth"), map_location="cpu"))
mlp.eval()
print("All models loaded successfully!")

df_test = pd.read_csv(os.path.join(mlp_dir, "sample_test_bursts.csv"))
feature_cols = mapping["features"]
classes = list(le.classes_)

X = df_test[feature_cols].values
X_scaled = scaler.transform(X)
y_str = df_test["modulation"].values
y_idx = le.transform(y_str)

# RF
t0 = time.perf_counter()
rf_probs = rf.predict_proba(X_scaled)
t_rf = (time.perf_counter() - t0) / len(X) * 1e6
rf_preds = np.argmax(rf_probs, axis=1)

# SVM
t0 = time.perf_counter()
svm_probs = svm.predict_proba(X_scaled)
t_svm = (time.perf_counter() - t0) / len(X) * 1e6
svm_preds = np.argmax(svm_probs, axis=1)

# MLP
X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
t0 = time.perf_counter()
with torch.no_grad():
    mlp_logits = mlp(X_tensor)
    mlp_probs = torch.softmax(mlp_logits, dim=1).numpy()
t_mlp = (time.perf_counter() - t0) / len(X) * 1e6
mlp_preds = np.argmax(mlp_probs, axis=1)

print("\n--- Latency per burst (Batch B=600) ---")
print(f"RF:  {t_rf:.2f} µs/burst")
print(f"SVM: {t_svm:.2f} µs/burst")
print(f"MLP: {t_mlp:.2f} µs/burst")

print("\n--- Overall Test (600 bursts) ---")
print(f"Random Forest : Acc = {accuracy_score(y_idx, rf_preds)*100:.2f}%, Macro-F1 = {f1_score(y_idx, rf_preds, average='macro')*100:.2f}%")
print(f"RBF-SVM       : Acc = {accuracy_score(y_idx, svm_preds)*100:.2f}%, Macro-F1 = {f1_score(y_idx, svm_preds, average='macro')*100:.2f}%")
print(f"3-Layer MLP   : Acc = {accuracy_score(y_idx, mlp_preds)*100:.2f}%, Macro-F1 = {f1_score(y_idx, mlp_preds, average='macro')*100:.2f}%")

snrs = sorted(df_test["snr_db"].unique())
rows = []
for s in snrs:
    mask = (df_test["snr_db"] == s).values
    y_s = y_idx[mask]
    rows.append({
        "SNR_dB": s,
        "Count": int(mask.sum()),
        "RF_Acc": round(accuracy_score(y_s, rf_preds[mask]) * 100, 2),
        "SVM_Acc": round(accuracy_score(y_s, svm_preds[mask]) * 100, 2),
        "MLP_Acc": round(accuracy_score(y_s, mlp_preds[mask]) * 100, 2),
        "RF_F1": round(f1_score(y_s, rf_preds[mask], average="macro", zero_division=0) * 100, 2),
        "SVM_F1": round(f1_score(y_s, svm_preds[mask], average="macro", zero_division=0) * 100, 2),
        "MLP_F1": round(f1_score(y_s, mlp_preds[mask], average="macro", zero_division=0) * 100, 2),
    })

df_res = pd.DataFrame(rows)
df_res.to_csv(os.path.join(mlp_dir, "models_benchmark_comparison.csv"), index=False)
print("\n--- Benchmark Table Across 15 SNRs ---")
print(df_res.to_string(index=False))
