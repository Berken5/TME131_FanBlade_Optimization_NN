import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import numpy as np

# ------------------ Load Data ------------------
df = pd.read_csv("AllRuns.csv")
df.columns = df.columns.str.strip()
df['Mass Flow [g/s]'] = df['Mass Flow [kg/s]'] * 1000

# Use only geometry-based inputs
X = df[['AoA', 'nBlades', 'BladeL']].values
y = df[['Mean Average Static Pressure [Pa]', 'Mass Flow [g/s]']].values
y_p = y[:, 0].reshape(-1, 1)
y_q = y[:, 1].reshape(-1, 1)

x_scaler = MinMaxScaler()
p_scaler = MinMaxScaler()
q_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_p_scaled = p_scaler.fit_transform(y_p)
y_q_scaled = q_scaler.fit_transform(y_q)
y_scaled = np.hstack((y_p_scaled, y_q_scaled))

joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(p_scaler, "p_scaler.save")
joblib.dump(q_scaler, "q_scaler.save")

X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=None
)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val_scaled, dtype=torch.float32)

# ------------------ Neural Network ------------------
class FanSplitNet(nn.Module):
    def __init__(self):
        super(FanSplitNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.pressure_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

        self.flow_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(128, 1),
            nn.Softplus()
        )

    def forward(self, x):
        shared_out = self.shared(x)
        p = self.pressure_head(shared_out)
        q = self.flow_head(shared_out)
        return torch.cat([p, q], dim=1)

# ------------------ Training ------------------
model = FanSplitNet()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)

    # Gradient-weighted loss mask for p only
    grad_weight = torch.ones_like(y_train[:, 0])
    grad_weight[1:] += torch.abs(y_train[:, 0][1:] - y_train[:, 0][:-1])
    loss_p = (grad_weight * torch.abs(preds[:, 0] - y_train[:, 0])).mean()

    loss_q = criterion(preds[:, 1], y_train[:, 1])
    loss = 0.6 * loss_p + 0.4 * loss_q

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss_p = criterion(val_preds[:, 0], y_val[:, 0])
        val_loss_q = criterion(val_preds[:, 1], y_val[:, 1])
        val_loss = 0.6 * val_loss_p + 0.4 * val_loss_q

    scheduler.step(val_loss)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "fan_model_split_heads.pth")

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

# ------------------ Plot ------------------
plt.figure(dpi=150)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss (scaled)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ Save Final Data ------------------
X_train_original = x_scaler.inverse_transform(X_train.numpy())
y_train_original = np.hstack((
    p_scaler.inverse_transform(y_train[:, 0].reshape(-1, 1)),
    q_scaler.inverse_transform(y_train[:, 1].reshape(-1, 1))
))
train_df = pd.DataFrame(X_train_original, columns=['AoA', 'nBlades', 'BladeL'])
train_df[['Mean Avg Static Pressure [Pa]', 'Mass Flow [g/s]']] = y_train_original
train_df.to_csv("training_data_used.csv", index=False)

X_val_original = x_scaler.inverse_transform(X_val.numpy())
y_val_original = np.hstack((
    p_scaler.inverse_transform(y_val[:, 0].reshape(-1, 1)),
    q_scaler.inverse_transform(y_val[:, 1].reshape(-1, 1))
))
val_df = pd.DataFrame(X_val_original, columns=['AoA', 'nBlades', 'BladeL'])
val_df[['Mean Avg Static Pressure [Pa]', 'Mass Flow [g/s]']] = y_val_original
val_df.to_csv("test_data_used.csv", index=False)

print("✅ Model trained and saved without early stopping")
