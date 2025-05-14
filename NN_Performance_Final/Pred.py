import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# -------- Neural Network Definition --------
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

# -------- Load Model and Scalers --------
model = FanSplitNet()
model.load_state_dict(torch.load("fan_model_split_heads.pth"))
model.eval()

x_scaler = joblib.load("x_scaler.save")
p_scaler = joblib.load("p_scaler.save")
q_scaler = joblib.load("q_scaler.save")

# --------- Single Point Prediction ---------
AOA = 65
nBlades = 36
BladeL = 0.005

input_data = np.array([[AOA, nBlades, BladeL]])
input_scaled = x_scaler.transform(input_data)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    output_scaled = model(input_tensor).numpy()
    p_unscaled = p_scaler.inverse_transform([[output_scaled[0, 0]]])[0, 0]
    q_unscaled = q_scaler.inverse_transform([[output_scaled[0, 1]]])[0, 0]

print(f"\n🔍 Prediction for AOA={AOA}, nBlades={nBlades}, BladeL={BladeL} m")
print(f"→ Predicted Static Pressure: {p_unscaled:.2f} Pa")
print(f"→ Predicted Mass Flow Rate: {q_unscaled:.2f} g/s")

# --------- Ground Truth Comparison ---------
try:
    df = pd.read_csv("AllRuns.csv")
    df.columns = df.columns.str.strip()
    df['Mass Flow [g/s]'] = df['Mass Flow [kg/s]'] * 1000

    match = df[(df["AoA"] == AOA) & (df["nBlades"] == nBlades) & (df["BladeL"] == BladeL)]
    if not match.empty:
        actual_p = match["Mean Average Static Pressure [Pa]"].values[0]
        actual_q = match["Mass Flow [g/s]"].values[0]
        print("\n✅ Found match in AllRuns.csv:")
        print(f"→ Actual Static Pressure: {actual_p:.2f} Pa")
        print(f"→ Actual Mass Flow Rate: {actual_q:.2f} g/s")
    else:
        print("\nℹ️ No exact match found in AllRuns.csv.")
except FileNotFoundError:
    print("\n⚠️ 'AllRuns.csv' not found.")

#%% --------- Grid-based Prediction + Plotting ---------
try:
    df = df[df["BladeL"] == BladeL]

    aoa_vals = np.linspace(df["AoA"].min(), df["AoA"].max(), 50)
    nblade_vals = np.linspace(df["nBlades"].min(), df["nBlades"].max(), 50)
    AOA_grid, NBLADES_grid = np.meshgrid(aoa_vals, nblade_vals)
    flat_grid = np.c_[AOA_grid.ravel(), NBLADES_grid.ravel(), np.full_like(AOA_grid.ravel(), BladeL)]

    inputs_scaled = x_scaler.transform(flat_grid)
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

    with torch.no_grad():
        outputs_scaled = model(inputs_tensor).numpy()

    pred_p = p_scaler.inverse_transform(outputs_scaled[:, 0].reshape(-1, 1)).reshape(AOA_grid.shape)
    pred_q = q_scaler.inverse_transform(outputs_scaled[:, 1].reshape(-1, 1)).reshape(AOA_grid.shape)

    true_p = griddata(df[["AoA", "nBlades"]].values, df["Mean Average Static Pressure [Pa]"].values, (AOA_grid, NBLADES_grid), method='linear')
    true_q = griddata(df[["AoA", "nBlades"]].values, df["Mass Flow [g/s]"].values, (AOA_grid, NBLADES_grid), method='linear')

    error_p = np.abs(pred_p - true_p) / np.abs(true_p) * 100
    error_q = np.abs(pred_q - true_q) / np.abs(true_q) * 100

    def plot_triplet(title, true_data, pred_data, error_data, unit):
        plt.rcParams['text.usetex'] = True
        plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 22,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16
    })

        fig, axs = plt.subplots(1, 3, figsize=(15, 5.0), dpi=300)
        vmin = np.nanmin([true_data, pred_data])
        vmax = np.nanmax([true_data, pred_data])

        # Panel 1: True
        im0 = axs[0].contourf(AOA_grid, NBLADES_grid, true_data, levels=50, cmap='plasma', vmin=vmin, vmax=vmax)
        axs[0].set_title(r"\textbf{Actual " + title + "}")
        cbar0 = fig.colorbar(im0, ax=axs[0])
        cbar0.ax.tick_params(labelsize=16)
        cbar0.set_label(r"\textbf{" + unit + "}", fontsize=18)

        # Panel 2: Predicted
        im1 = axs[1].contourf(AOA_grid, NBLADES_grid, pred_data, levels=50, cmap='plasma', vmin=vmin, vmax=vmax)
        axs[1].set_title(r"\textbf{Predicted " + title + "}")
        cbar1 = fig.colorbar(im1, ax=axs[1])
        cbar1.ax.tick_params(labelsize=16)
        cbar1.set_label(r"\textbf{" + unit + "}", fontsize=18)

        # Panel 3: Error
        im2 = axs[2].contourf(AOA_grid, NBLADES_grid, error_data, levels=50, cmap='Greys')
        axs[2].set_title(r"\textbf{Error in " + title + "}")
        cbar2 = fig.colorbar(im2, ax=axs[2])
        cbar2.ax.tick_params(labelsize=16)
        cbar2.set_label(r"\textbf{Percentage Error}", fontsize=18)

    # Axis labels
        for ax in axs:
            ax.set_xlabel(r"\textbf{AoA [°]}")
            ax.set_ylabel(r"\textbf{nBlades}")

    # Suptitle
        plt.suptitle(r"\textbf{" + title + " Prediction vs Actual}")
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}_triplet.png", dpi=300, bbox_inches='tight')
        plt.show()

    plot_triplet("Static Pressure", true_p, pred_p, error_p, "Pa")
    plot_triplet("Mass Flow Rate", true_q, pred_q, error_q, "g/s")


except Exception as e:
    print(f"\n⚠️ Error during grid prediction/plotting: {e}")
