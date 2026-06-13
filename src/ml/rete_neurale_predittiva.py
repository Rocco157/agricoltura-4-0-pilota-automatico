import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Caricamento e preparazione ===
data = pd.read_csv("agricultural_field_path_planning_dataset.csv")

# === 2. Selezione delle feature e label ===
features = [
    'original_area_area', 'original_area_perimeter',
    'original_area_width', 'original_area_height', 'original_area_aspect_ratio',
    'num_vertices_area', 'num_obstacles', 'total_obstacle_area', 'obstacle_area_ratio',
    'area_capezzagne_ostacoli', 'area_capezzagne_bordocampo', 'num_decomposed_cells',
    'total_trajectory_length'
]
labels_reg = ['label_orientation_x', 'label_orientation_y']
label_clf = ['field_type']

# Mappatura field_type a interi per classificazione
field_type_map = {label: idx for idx, label in enumerate(data['field_type'].unique())}
reverse_map = {v: k for k, v in field_type_map.items()}
data['algo_class'] = data['field_type'].map(field_type_map)

X = data[features].fillna(0)
y_reg = data[labels_reg].fillna(0)
y_clf = data['algo_class'].fillna(0).astype(int)

# === 3. Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Dataset e Dataloader ===
class FieldDataset(Dataset):
    def __init__(self, X, y_reg, y_clf):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg.values, dtype=torch.float32)
        self.y_clf = torch.tensor(y_clf.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_clf[idx]

X_train, X_val, yreg_train, yreg_val, yclf_train, yclf_val = train_test_split(
    X_scaled, y_reg, y_clf, test_size=0.2, random_state=42
)
train_dataset = FieldDataset(X_train, yreg_train, yclf_train)
val_dataset = FieldDataset(X_val, yreg_val, yclf_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === 5. Definizione modello combinato ===
class FieldNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FieldNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(64, 2)
        self.clf_head = nn.Linear(64, num_classes)

    def forward(self, x):
        shared = self.shared(x)
        return self.reg_head(shared), self.clf_head(shared)

model = FieldNet(input_size=X.shape[1], num_classes=len(field_type_map))
reg_criterion = nn.MSELoss()
clf_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 6. Training loop ===
epochs = 100
for epoch in range(epochs):
    model.train()
    total_reg_loss, total_clf_loss = 0, 0

    for xb, yb_reg, yb_clf in train_loader:
        pred_reg, pred_clf = model(xb)
        loss_reg = reg_criterion(pred_reg, yb_reg)
        loss_clf = clf_criterion(pred_clf, yb_clf)
        loss = loss_reg + loss_clf

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reg_loss += loss_reg.item()
        total_clf_loss += loss_clf.item()

    model.eval()
    val_reg_loss, val_clf_loss = 0, 0
    y_true, y_pred = [], []
    reg_true, reg_pred = [], []

    with torch.no_grad():
        for xb, yb_reg, yb_clf in val_loader:
            pred_reg, pred_clf = model(xb)
            val_reg_loss += reg_criterion(pred_reg, yb_reg).item()
            val_clf_loss += clf_criterion(pred_clf, yb_clf).item()

            y_true.extend(yb_clf.tolist())
            y_pred.extend(torch.argmax(pred_clf, dim=1).tolist())
            reg_true.extend(yb_reg.tolist())
            reg_pred.extend(pred_reg.tolist())

    print(f"Epoch {epoch+1}/{epochs} - Reg Loss: {total_reg_loss/len(train_loader):.4f} - Clf Loss: {total_clf_loss/len(train_loader):.4f} - Val Reg Loss: {val_reg_loss/len(val_loader):.4f} - Val Clf Loss: {val_clf_loss/len(val_loader):.4f}")

# === 7. Salvataggio ===
torch.save(model.state_dict(), "field_net_multi_task.pt")

# === 8. Valutazione finale ===
acc = accuracy_score(y_true, y_pred)
mse = mean_squared_error(reg_true, reg_pred)
print(f"\n[Valutazione finale] Accuracy classificazione: {acc:.4f}, MSE regressione: {mse:.4f}")

# === 9. Confusion matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(field_type_map.keys()),
            yticklabels=list(field_type_map.keys()))
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Confusion Matrix - Strategia Algoritmica")
plt.tight_layout()
plt.show()
