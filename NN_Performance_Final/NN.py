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
    X_scaled, y_scaled, test_size=0.2, random_state=42
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
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.pressure_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

        self.flow_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 1),
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
val_losses_p=[]
val_losses_q=[]


train_losses_p = []
train_losses_q = []

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)

    # Gradient-weighted loss mask for p only
    # grad_weight = torch.ones_like(y_train[:, 0])
    # grad_weight[1:] += torch.abs(y_train[:, 0][1:] - y_train[:, 0][:-1])
    # loss_p = (grad_weight * torch.abs(preds[:, 0] - y_train[:, 0])).mean()
    
    loss_p=criterion(preds[:, 0], y_train[:, 0])
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
    train_losses_p.append(loss_p.item())
    train_losses_q.append(loss_q.item())

    val_losses.append(val_loss.item())
    val_losses_p.append(val_loss_p.item())
    val_losses_q.append(val_loss_q.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "fan_model_split_heads.pth")

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

#%% ------------------ Plot ------------------

plt.rcParams.update({
        # 'font.family': 'Times New Roman',
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        # 'legend.fontsize': 16
    })


plt.figure(dpi=300,figsize=(5,4))
plt.plot(train_losses,'k-', label="Train Loss")
plt.plot(val_losses,'g-' ,label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.show()

plt.figure(dpi=300,figsize=(5,4))
plt.plot(train_losses_p,color='red', label="Train Pressure Loss")
plt.plot(val_losses_p,color='hotpink' ,label="Validation Pressure Loss")
plt.plot(train_losses_q,color='blue', label="Train Mass Flow Loss")
plt.plot(val_losses_q,color='darkturquoise' ,label="Validation Mass Flow Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Individual Loss Components")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=300)

# First subplot: Overall Train and Validation Loss
axs[0].plot(train_losses, 'k-', label="Train Loss")
axs[0].plot(val_losses, 'g-', label="Validation Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("MAE Loss")
axs[0].set_title("Training and Validation Loss")
axs[0].legend()
axs[0].grid(linestyle='--')

# Second subplot: Individual Loss Components
axs[1].plot(train_losses_p, color='red', label="Train Pressure Loss")
axs[1].plot(val_losses_p, color='hotpink', label="Validation Pressure Loss")
axs[1].plot(train_losses_q, color='blue', label="Train Mass Flow Loss")
axs[1].plot(val_losses_q, color='darkturquoise', label="Validation Mass Flow Loss")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MAE Loss")
axs[1].set_title("Individual Loss Components")
axs[1].legend()
axs[1].grid(linestyle='--')
# plt.tight_layout()
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

