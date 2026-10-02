import json
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "hand_seal_dataset.csv")

# Load Dataset
df = pd.read_csv(CSV_PATH)
feature_cols = [
    c for c in df.columns if c.startswith("L_") or c.startswith("R_")
]

X_raw = df[feature_cols].values
y_raw = df["seal"].values

# Presence Flags
has_l = (X_raw[:, 3] != 0).astype(np.float32).reshape(-1, 1)
has_r = (X_raw[:, 66] != 0).astype(np.float32).reshape(-1, 1)
X_augmented = np.hstack([X_raw, has_l, has_r])

# Encode & Scale
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_augmented)

# DataLoader for proper BatchNorm statistics
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y_encoded)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Neural Network Model
class HandSealNN(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(HandSealNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


num_classes = len(label_encoder.classes_)
input_dim = X_scaled.shape[1]

model = HandSealNN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training Loop
print("Training Neural Network with Mini-Batches...")
epochs = 60
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for bx, by in loader:
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}] | Loss: {total_loss/len(loader):.4f}")

# Save Model & Artifacts
torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "seal_nn_model.pth"))
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "nn_scaler.pkl"))
joblib.dump(label_encoder, os.path.join(SCRIPT_DIR, "label_encoder.pkl"))

metadata = {
    "input_dim": input_dim,
    "feature_columns": feature_cols,
    "classes": label_encoder.classes_.tolist(),
}
with open(os.path.join(SCRIPT_DIR, "nn_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\nRetraining complete! Model saved successfully.")