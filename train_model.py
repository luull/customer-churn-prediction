import os
import random
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
DATA_URL = "https://raw.githubusercontent.com/SohelRaja/Customer-Churn-Analysis/master/Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "InternetService", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
MODEL_PATH = "model/customer_churn_model.pth"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def prepare_features(frame, metadata=None):
    """Membersihkan data dan mengubah kategori menjadi one-hot secara konsisten."""
    x = frame[FEATURES].copy()

    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if metadata is None:
        medians = {c: float(x[c].median()) for c in NUMERIC}
        category_values = {}
        category_modes = {}
        for col in CATEGORICAL:
            mode = x[col].mode(dropna=True)
            category_modes[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
            x[col] = x[col].fillna(category_modes[col]).astype(str)
            category_values[col] = sorted(x[col].unique().tolist())
        means = {}
        stds = {}
        for col in NUMERIC:
            x[col] = x[col].fillna(medians[col])
            means[col] = float(x[col].mean())
            std = float(x[col].std())
            stds[col] = std if std > 0 else 1.0
        metadata = {
            "numeric_medians": medians,
            "numeric_means": means,
            "numeric_stds": stds,
            "category_values": category_values,
            "category_modes": category_modes,
            "numeric_columns": NUMERIC,
            "categorical_columns": CATEGORICAL,
        }
    else:
        for col in NUMERIC:
            x[col] = x[col].fillna(metadata["numeric_medians"][col])
        for col in CATEGORICAL:
            x[col] = x[col].fillna(metadata["category_modes"][col]).astype(str)

    numeric_array = np.column_stack([
        (x[col].astype(float).to_numpy() - metadata["numeric_means"][col])
        / metadata["numeric_stds"][col]
        for col in NUMERIC
    ]).astype(np.float32)

    categorical_arrays = []
    for col in CATEGORICAL:
        categories = metadata["category_values"][col]
        values = x[col].astype(str).to_numpy()
        one_hot = np.zeros((len(x), len(categories)), dtype=np.float32)
        lookup = {value: idx for idx, value in enumerate(categories)}
        for row_idx, value in enumerate(values):
            if value in lookup:
                one_hot[row_idx, lookup[value]] = 1.0
        categorical_arrays.append(one_hot)

    encoded = np.concatenate([numeric_array] + categorical_arrays, axis=1)
    return encoded, metadata

class ChurnNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

def main():
    set_seed()
    os.makedirs("model", exist_ok=True)

    print("Mengunduh dataset Telco Customer Churn...")
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as exc:
        raise RuntimeError(
            "Dataset gagal diunduh. Periksa koneksi internet/sertifikat SSL, "
            "atau unduh CSV secara manual dan sesuaikan DATA_URL."
        ) from exc

    data = df[FEATURES + ["Churn"]].copy()
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = data.dropna(subset=["Churn"])
    X = data[FEATURES]
    y = data["Churn"].astype(np.float32).to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, metadata = prepare_features(X_train_df)
    X_test, _ = prepare_features(X_test_df, metadata)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = ChurnNeuralNetwork(X_train.shape[1])
    positive = max(float(y_train.sum()), 1.0)
    negative = max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Sisihkan sebagian data training untuk validasi.
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.15,
        random_state=SEED, stratify=y_train
    )
    best_val_loss = float("inf")
    best_state = None
    patience, patience_count = 12, 0
    epochs = 100
    batch_size = 128

    print("Mulai training model PyTorch...")
    for epoch in range(epochs):
        model.train()
        shuffled = np.random.permutation(train_idx)
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start:start + batch_size]
            logits = model(X_train_tensor[batch_idx])
            loss = criterion(logits, y_train_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_train_tensor[val_idx])
            val_loss = criterion(val_logits, y_train_tensor[val_idx]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - validation loss: {val_loss:.4f}")
        if patience_count >= patience:
            print(f"Early stopping pada epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test_tensor)).numpy()
    predictions = (probabilities >= 0.5).astype(int)

    print("\nEvaluasi data testing:")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(X_train.shape[1]),
        "metadata": metadata,
        "threshold": 0.5,
        "model_name": "Customer Churn Neural Network (PyTorch)",
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"\nModel berhasil disimpan: {MODEL_PATH}")

if __name__ == "__main__":
    main()
