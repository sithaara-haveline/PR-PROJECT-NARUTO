import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Dataset
df = pd.read_csv("hand_seal_dataset.csv")

feature_cols = [
    c for c in df.columns if c.startswith("L_") or c.startswith("R_")
]
X_raw = df[feature_cols].values
y_raw = df["seal"].values


def add_presence_features(X_matrix):
    has_l = (X_matrix[:, 3] != 0).astype(np.float32).reshape(-1, 1)
    has_r = (X_matrix[:, 66] != 0).astype(np.float32).reshape(-1, 1)
    return np.hstack([X_matrix, has_l, has_r])


X_augmented = add_presence_features(X_raw)

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# 2. Sequential Block Split (Prevents adjacent video frame leakage)
# Take 80% chunk per class for training, last 20% chunk for testing
train_idx, test_idx = [], []
for c in np.unique(y_encoded):
    c_indices = np.where(y_encoded == c)[0]
    split_pt = int(len(c_indices) * 0.8)
    train_idx.extend(c_indices[:split_pt])
    test_idx.extend(c_indices[split_pt:])

X_train, X_test = X_augmented[train_idx], X_augmented[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Model with Heavy Regularization
class HandSealNN(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(HandSealNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train_scaled.shape[1]
num_classes = len(label_encoder.classes_)

model = HandSealNN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
# Added weight_decay (L2 regularization) to penalize memorization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.LongTensor(y_test)

print("Training Regularized Neural Network...")
epochs = 120

for epoch in range(epochs):
    model.train()

    # Add Gaussian Noise Augmentation to Training Features
    noise = torch.randn_like(X_train_t) * 0.05
    noisy_X_train = X_train_t + noise

    optimizer.zero_grad()
    outputs = model(noisy_X_train)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            acc = (
                (test_outputs.argmax(dim=1) == y_test_t)
                .float()
                .mean()
                .item()
            )
            print(
                f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f} | Test Acc: {acc*100:.2f}%"
            )

# Save artifacts
torch.save(model.state_dict(), "seal_nn_model.pth")
joblib.dump(scaler, "nn_scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

metadata = {
    "feature_columns": feature_cols,
    "classes": list(label_encoder.classes_),
    "input_dim": input_dim,
}
with open("nn_metadata.json", "w") as f:
    json.dump(metadata, f)

print("Model successfully trained with block splitting & noise augmentation!")