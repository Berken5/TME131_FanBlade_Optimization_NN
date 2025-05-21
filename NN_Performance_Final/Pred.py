import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator


# -------- Neural Network Definition --------
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

# -------- Load Model and Scalers --------
model = FanSplitNet()
model.load_state_dict(torch.load("fan_model_split_heads.pth"))
model.eval()

x_scaler = joblib.load("x_scaler.save")
p_scaler = joblib.load("p_scaler.save")
q_scaler = joblib.load("q_scaler.save")

#%% --------- Single Point Prediction ---------
AOA = 45
nBlades = 28
BladeL = 0.005

input_data = np.array([[AOA, nBlades, BladeL]])
input_scaled = x_scaler.transform(input_data)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    output_scaled = model(input_tensor).numpy()
    p_unscaled = p_scaler.inverse_transform([[output_scaled[0, 0]]])[0, 0]
    q_unscaled = q_scaler.inverse_transform([[output_scaled[0, 1]]])[0, 0]

print(f"\n Prediction for AOA={AOA}, nBlades={nBlades}, BladeL={BladeL} m")
print(f"→ Predicted Static Pressure: {p_unscaled:.2f} Pa")
print(f"→ Predicted Mass Flow Rate: {q_unscaled:.2f} g/s")

# --------- Ground Truth Comparison ---------
try:
    df = pd.read_csv("data/AllRuns_3.csv")
    df.columns = df.columns.str.strip()
    df['Mass Flow [g/s]'] = df['Mass Flow [kg/s]'] * 1000

    match = df[(df["AoA"] == AOA) & (df["nBlades"] == nBlades) & (df["BladeL"] == BladeL)]
    if not match.empty:
        actual_p = match["Mean Average Static Pressure [Pa]"].values[0]
        actual_q = match["Mass Flow [g/s]"].values[0]
        print("\n Found match in AllRuns.csv:")
        print(f"→ Simulated Static Pressure: {actual_p:.2f} Pa")
        print(f"→ Simulated Mass Flow Rate: {actual_q:.2f} g/s")
        
        error_pre = np.abs((actual_p-p_unscaled)/actual_p)*100
        error_mass = np.abs(actual_q-q_unscaled)/actual_q*100
        print("\n Error:")
        print(f"→ Error in Static Pressure: {error_pre:.2f}")
        print(f"→ Error in Mass Flow Rate: {error_mass:.2f}")

        print("\n MAE:")
        print(f"→ MAE Static Pressure: {np.abs(actual_p-p_unscaled):.2f}")
        print(f"→ MAE Mass Flow Rate: {np.abs(actual_q-q_unscaled):.2f}")

        print("\n MSE:")
        print(f"→ MSE Static Pressure: {(actual_p-p_unscaled)**2:.3f}")
        print(f"→ MSE Mass Flow Rate: {(actual_q-q_unscaled)**2:.3f}")
    else:
        print("\nℹ No exact match found in AllRuns.csv.")
except FileNotFoundError:
    print("\n 'AllRuns.csv' not found.")


#%% --------- Grid-based Prediction + Plotting ---------
df = pd.read_csv("AllRuns.csv")
df.columns = df.columns.str.strip()
df['Mass Flow [g/s]'] = df['Mass Flow [kg/s]'] * 1000

match = df[(df["AoA"] == AOA) & (df["nBlades"] == nBlades) & (df["BladeL"] == BladeL)]
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
        # plt.rcParams['text.usetex'] = True
        plt.rcParams.update({
        # 'font.family': 'Times New Roman',
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        # 'legend.fontsize': 16
    })
    

        fig, axs = plt.subplots(1, 3, figsize=(14, 2.8), dpi=300, gridspec_kw={"width_ratios": [1.1, 1.1, 1], "wspace": 0.25})
        vmin = np.nanmin([true_data, pred_data])
        vmax = np.nanmax([true_data, pred_data])

        # cmap_main = 'viridis' if "Static Pressure" in title else 'gnuplot'
# Panel 1: True
        im0 = axs[0].contourf(AOA_grid, NBLADES_grid, true_data, levels=50, cmap='plasma', vmin=vmin, vmax=vmax)
        axs[0].set_title("CFD " + title)

# Panel 2: Predicted
        im1 = axs[1].contourf(AOA_grid, NBLADES_grid, pred_data, levels=50, cmap='plasma', vmin=vmin, vmax=vmax)
        axs[1].set_title("Predicted " + title)

# Panel 3: Error
        im2 = axs[2].contourf(AOA_grid, NBLADES_grid, error_data, levels=50, cmap='Greys')
        axs[2].set_title("Error in " + title)

        cbar0 = fig.colorbar(im0, ax=[axs[0], axs[1]], location='left', pad=0.11, shrink=0.9)
        cbar0.set_label(unit, labelpad=8)  # Increase space between label and colorbar
        cbar0.ax.yaxis.label.set_rotation(90)
        cbar0.locator = MaxNLocator(nbins=5)
        cbar0.ax.yaxis.set_label_position('left')

# Right colorbar (for error)
        cbar2 = fig.colorbar(im2, ax=axs[2], pad=0.05, shrink=0.9)
        cbar2.ax.tick_params()
        cbar2.set_label("Percentage Error")
        cbar2.locator = MaxNLocator(nbins=5)
        cbar2.update_ticks()

# Labels
        for ax in axs:
            ax.set_xlabel("AoA [°]")
            ax.set_ylabel("nBlades")
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10

# Remove redundant y-labels
        axs[1].set_ylabel("")
        axs[2].set_ylabel("")

        # plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}_triplet.png", dpi=300)
        plt.show()

    plot_triplet("Static Pressure", true_p, pred_p, error_p, "Pressure [Pa]")
    plot_triplet("Mass Flow Rate", true_q, pred_q, error_q, "Mass Flow [g/s]")


except Exception as e:
    print(f"\n Error during grid prediction/plotting: {e}")
