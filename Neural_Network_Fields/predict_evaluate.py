import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from preprocessing import FluidGridInterpolator
from Fanblade_fields_NN import FanBladeNN, extract_parameters_from_filename

plt.rcParams.update({
    'font.size': 13,            # Default text size
    'axes.titlesize': 14,       # Axes title size
    'axes.labelsize': 13,       # Axes label size
    'xtick.labelsize': 13,      # X tick label size
    'ytick.labelsize': 13,      # Y tick label size
})

def load_model_and_scalers(model_path, scalers_path, device):
    scalers = joblib.load(scalers_path)
    num_inputs = 7  # 3 params + 3 coords + 1 is_fluid
    model = FanBladeNN(num_inputs).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model, scalers


# Loss metrics
def mse_loss_fluid(pred_pressure, pred_velocity, target_pressure, target_velocity, mask, pressure_weight=1.0, velocity_weight=1.0):
    """Compute Mean Squared Error, only on fluid nodes"""
    pressure_loss = (pred_pressure - target_pressure) ** 2
    velocity_loss = (pred_velocity - target_velocity) ** 2
    pressure_loss = pressure_loss * mask.unsqueeze(-1)
    velocity_loss = velocity_loss * mask.unsqueeze(-1)
    mean_pressure_loss = pressure_loss.sum() / (mask.sum() + 1e-30)
    mean_velocity_loss = velocity_loss.sum() / (mask.sum() * 3 + 1e-30)
    total_loss = pressure_weight * mean_pressure_loss + velocity_weight * mean_velocity_loss
    return total_loss, mean_pressure_loss, mean_velocity_loss

def mae_loss_fluid(pred_pressure, pred_velocity, target_pressure, target_velocity, mask, pressure_weight=1.0, velocity_weight=1.0):
    """Compute Mean Absolute Error, only on fluid nodes"""
    pressure_loss = torch.abs(pred_pressure - target_pressure)
    velocity_loss = torch.abs(pred_velocity - target_velocity)
    pressure_loss = pressure_loss * mask.unsqueeze(-1)
    velocity_loss = velocity_loss * mask.unsqueeze(-1)
    mean_pressure_loss = pressure_loss.sum() / (mask.sum() + 1e-30)
    mean_velocity_loss = velocity_loss.sum() / (mask.sum() * 3 + 1e-30)
    total_loss = pressure_weight * mean_pressure_loss + velocity_weight * mean_velocity_loss
    return total_loss, mean_pressure_loss, mean_velocity_loss

def r2_score_fluid(pred_pressure, pred_velocity, target_pressure, target_velocity, mask, pressure_weight=1.0, velocity_weight=1.0):
    """Compute R2 score, only on fluid nodes"""
    # Flatten mask for indexing
    mask_bool = mask == 1
    # Pressure
    y_true_p = target_pressure[mask_bool]
    y_pred_p = pred_pressure[mask_bool]
    # Velocity
    y_true_v = target_velocity[mask_bool]
    y_pred_v = pred_velocity[mask_bool]
    # R2 for pressure
    ss_res_p = ((y_true_p - y_pred_p) ** 2).sum()
    ss_tot_p = ((y_true_p - y_true_p.mean()) ** 2).sum()
    r2_p = 1 - ss_res_p / (ss_tot_p + 1e-30)
    # R2 for velocity (mean over 3 components)
    ss_res_v = ((y_true_v - y_pred_v) ** 2).sum(dim=0)
    ss_tot_v = ((y_true_v - y_true_v.mean(dim=0)) ** 2).sum(dim=0)
    r2_v = 1 - ss_res_v / (ss_tot_v + 1e-30)
    r2_v_mean = r2_v.mean()
    # Weighted total
    r2_total = (r2_p + r2_v_mean) / 2
    return r2_total.item(), r2_p.item(), r2_v_mean.item()


def predict(model, file_path, all_nodes_file, scalers, device, pressure_weight=1.0, velocity_weight=1.0):
    """Make predictions for a simulation file and compute loss"""
    params = extract_parameters_from_filename(os.path.basename(file_path))
    params_scaled = scalers['param_scaler'].transform(np.array([params]).astype(np.float32)).flatten()

    processor = FluidGridInterpolator(file_path, all_nodes_file)
    df = processor.process()
    num_nodes = len(df)

    X = np.zeros((num_nodes, 7), dtype=np.float32)
    X[:, 0:3] = params_scaled
    X[:, 3:6] = scalers['spatial_scaler'].transform(df[['X (m)', 'Y (m)', 'Z (m)']].values)
    X[:, 6] = df['is_fluid'].values

    Y = np.zeros((num_nodes, 4), dtype=np.float32)
    Y[:, 0] = scalers['pressure_scaler'].transform(df['Pressure (Pa)'].values.reshape(-1, 1)).flatten()
    Y[:, 1:4] = scalers['velocity_scaler'].transform(df[['Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)']].values)

    mask = df['is_fluid'].values.astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pressure_pred, velocity_pred = model(X_tensor)
        # MSE
        loss, p_loss, v_loss = mse_loss_fluid(
            pressure_pred, velocity_pred,
            Y_tensor[:, 0:1], Y_tensor[:, 1:4],
            mask_tensor,
            pressure_weight=pressure_weight, velocity_weight=velocity_weight
        )
        # MAE
        mae_total, mae_pressure, mae_velocity = mae_loss_fluid(
            pressure_pred, velocity_pred,
            Y_tensor[:, 0:1], Y_tensor[:, 1:4],
            mask_tensor,
            pressure_weight=pressure_weight, velocity_weight=velocity_weight
        )
        # R2
        r2_total, r2_pressure, r2_velocity = r2_score_fluid(
            pressure_pred, velocity_pred,
            Y_tensor[:, 0:1], Y_tensor[:, 1:4],
            mask_tensor,
            pressure_weight=pressure_weight, velocity_weight=velocity_weight
        )

        # Inverse transform predictions for output
        pressure_pred_np = pressure_pred.cpu().numpy()
        velocity_pred_np = velocity_pred.cpu().numpy()
        pressure_pred_np = scalers['pressure_scaler'].inverse_transform(pressure_pred_np)
        velocity_pred_np = scalers['velocity_scaler'].inverse_transform(velocity_pred_np)

        output_df = df.copy()
        output_df['Pressure_Pred (Pa)'] = pressure_pred_np
        output_df['Velocity[i]_Pred (m/s)'] = velocity_pred_np[:, 0]
        output_df['Velocity[j]_Pred (m/s)'] = velocity_pred_np[:, 1]
        output_df['Velocity[k]_Pred (m/s)'] = velocity_pred_np[:, 2]

    # Return all metrics and the DataFrame
    return (output_df,
            loss.item(), p_loss.item(), v_loss.item(),
            mae_total.item(), mae_pressure.item(), mae_velocity.item(),
            r2_total, r2_pressure, r2_velocity)


def velocity_magnitude(df):
    # Real velocity magnitude
    df['Real_Velocity_Mag (m/s)'] = np.linalg.norm(
        df[['Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)']].values, axis=1
    )
    # Predicted velocity magnitude
    df['Pred_Velocity_Mag (m/s)'] = np.linalg.norm(
        df[['Velocity[i]_Pred (m/s)', 'Velocity[j]_Pred (m/s)', 'Velocity[k]_Pred (m/s)']].values, axis=1
    )
    return df

def prediction_error(df):
    # Error in pressure
    df['Pressure_Percent_Diff'] = df['Pressure_Pred (Pa)'] - df['Pressure (Pa)']

    # Error in velocity magnitude
    df['Velocity_Percent_Diff'] = df['Pred_Velocity_Mag (m/s)'] - df['Real_Velocity_Mag (m/s)']

    return df

def plot_comparison(df, simulation_filename, plot_solids=False):
    fluid_mask = df['is_fluid'] == 1
    solid_mask = df['is_fluid'] == 0

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), constrained_layout=True)

    # Set vmin/vmax based on real data
    pressure_min = df['Pressure (Pa)'].min()
    pressure_max = df['Pressure (Pa)'].max()
    vel_min = df['Real_Velocity_Mag (m/s)'].min()
    vel_max = df['Real_Velocity_Mag (m/s)'].max()

    # Real Pressure
    sc0 = axes[0, 0].scatter(
        df.loc[fluid_mask, 'X (m)'], df.loc[fluid_mask, 'Y (m)'],
        c=df.loc[fluid_mask, 'Pressure (Pa)'], cmap='viridis', s=1, marker='s',
        vmin=pressure_min, vmax=pressure_max, label='Fluid Nodes'
    )
    if plot_solids==True:
        sc0 = axes[0, 0].scatter(
            df.loc[solid_mask, 'X (m)'], df.loc[solid_mask, 'Y (m)'],
            c=df.loc[solid_mask, 'Pressure (Pa)'], cmap='viridis', s=1, marker='o',
            vmin=pressure_min, vmax=pressure_max, label='Solid Nodes'
        )
        axes[0, 0].legend()
    axes[0, 0].set_title('CFD Pressure')
    axes[0, 0].axis('equal')

    # Predicted Pressure
    sc1 = axes[0, 1].scatter(
        df.loc[fluid_mask, 'X (m)'], df.loc[fluid_mask, 'Y (m)'],
        c=df.loc[fluid_mask, 'Pressure_Pred (Pa)'], cmap='viridis', s=1, marker='s',
        vmin=pressure_min, vmax=pressure_max, label='Fluid Nodes'
    )
    if plot_solids==True:
        sc1 = axes[0, 1].scatter(
            df.loc[solid_mask, 'X (m)'], df.loc[solid_mask, 'Y (m)'],
            c=df.loc[solid_mask, 'Pressure_Pred (Pa)'], cmap='viridis', s=1, marker='o',
            vmin=pressure_min, vmax=pressure_max, label='Solid Nodes'
        )
        axes[0, 1].legend()
    axes[0, 1].set_title('Predicted Pressure')
    axes[0, 1].axis('equal')

    # --- SHARED COLORBAR FOR PRESSURE ---
    cbar0 = plt.colorbar(sc1, ax=[axes[0, 0], axes[0, 1]], orientation='vertical', location='left', pad=0.01, fraction=0.046)
    cbar0.set_label('Pressure [Pa]')


    # Real Velocity Magnitude
    sc2 = axes[1, 0].scatter(
        df.loc[fluid_mask, 'X (m)'], df.loc[fluid_mask, 'Y (m)'],
        c=df.loc[fluid_mask, 'Real_Velocity_Mag (m/s)'], cmap='gnuplot', s=1, marker='s',
        vmin=vel_min, vmax=vel_max, label='Fluid Nodes'
    )
    if plot_solids==True:
        sc2 = axes[1, 0].scatter(
            df.loc[solid_mask, 'X (m)'], df.loc[solid_mask, 'Y (m)'],
            c=df.loc[solid_mask, 'Real_Velocity_Mag (m/s)'], cmap='gnuplot', s=1, marker='o',
            vmin=vel_min, vmax=vel_max, label='Solid Nodes'
        )
        axes[1, 0].legend()
    axes[1, 0].set_title('CFD Velocity Magnitude')
    axes[1, 0].axis('equal')

    # Predicted Velocity Magnitude
    sc3 = axes[1, 1].scatter(
        df.loc[fluid_mask, 'X (m)'], df.loc[fluid_mask, 'Y (m)'],
        c=df.loc[fluid_mask, 'Pred_Velocity_Mag (m/s)'], cmap='gnuplot', s=1, marker='s',
        vmin=vel_min, vmax=vel_max, label='Fluid Nodes'
    )
    if plot_solids==True:
        sc3 = axes[1, 1].scatter(
            df.loc[solid_mask, 'X (m)'], df.loc[solid_mask, 'Y (m)'],
            c=df.loc[solid_mask, 'Pred_Velocity_Mag (m/s)'], cmap='gnuplot', s=1, marker='o',
            vmin=vel_min, vmax=vel_max, label='Solid Nodes'
        )
        axes[1, 1].legend()
    axes[1, 1].set_title('Predicted Velocity Magnitude')
    axes[1, 1].axis('equal')

    # --- SHARED COLORBAR FOR VELOCITY ---
    cbar1 = fig.colorbar(sc3, ax=[axes[1, 0], axes[1, 1]], orientation='vertical', location='left', pad=0.01, fraction=0.046)
    cbar1.set_label('Velocity [m/s]')
    

    # Prediction error for pressure
    sc4 = axes[0, 2].scatter(
        df.loc[fluid_mask, 'X (m)'], df.loc[fluid_mask, 'Y (m)'],
        c=df.loc[fluid_mask, 'Pressure_Percent_Diff'], cmap='RdBu', s=1, marker='s',
        vmin=-20, vmax=20, label='Fluid Nodes'
        )
    if plot_solids==True:
        sc4 = axes[0, 2].scatter(
            df.loc[solid_mask, 'X (m)'], df.loc[solid_mask, 'Y (m)'],
            c=df.loc[solid_mask, 'Pressure_Percent_Diff'], cmap='RdBu', s=1, marker='o',
            vmin=-20, vmax=20, label='Solid Nodes'
        )
        axes[0, 2].legend()
    plt.colorbar(sc4, ax=axes[0, 2], label='Pressure [Pa]')
    axes[0, 2].set_title('Prediction Error Pressure')
    axes[0, 2].axis('equal')
    

    # Prediction error for velocity magnitude
    sc5 = axes[1, 2].scatter(
        df.loc[fluid_mask, 'X (m)'], df.loc[fluid_mask, 'Y (m)'],
        c=df.loc[fluid_mask, 'Velocity_Percent_Diff'], cmap='RdBu', s=1, marker='s',
        vmin=-5, vmax=5, label='Fluid Nodes'
        )
    if plot_solids==True:
        sc5 = axes[1, 2].scatter(
            df.loc[solid_mask, 'X (m)'], df.loc[solid_mask, 'Y (m)'],
            c=df.loc[solid_mask, 'Velocity_Percent_Diff'], cmap='RdBu', s=1, marker='o',
            vmin=-5, vmax=5, label='Solid Nodes'
        )
        axes[1, 2].legend()
    plt.colorbar(sc5, ax=axes[1, 2], label='Velocity [m/s]')
    axes[1, 2].set_title('Prediction Error Velocity')
    axes[1, 2].axis('equal')
    
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(simulation_filename))[0]
    plt.savefig(f"Plots/{file_name}_plots_3x2.png", dpi=300)
    #plt.show()


if __name__ == "__main__":
    simfile = "simfiles_true_validation/55_30_0.0050.csv"
    all_nodes = "all_nodes.csv"
    model_path = "checkpoints/best_checkpoint.pt"
    scalers_path = "checkpoints/scalers.pkl"
    output_csv = '55_30_0.0050_predicted.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and scalers
    model, scalers = load_model_and_scalers(model_path, scalers_path, device)

    # Predict on the input file
    df_pred, total_loss, pressure_loss, velocity_loss, mae_total, mae_pressure, mae_velocity, r2_total, r2_pressure, r2_velocity = predict(
        model, simfile, all_nodes, scalers, device
    )

    print(f"Total MSE: {total_loss:.3f}, Pressure MSE: {pressure_loss:.3f}, Velocity MSE: {velocity_loss:.3f}")
    print(f"Total MAE: {mae_total:.3f}, Pressure MAE: {mae_pressure:.3f}, Velocity MAE: {mae_velocity:.3f}")
    print(f"Total R2: {r2_total:.2f}, Pressure R2: {r2_pressure:.2f}, Velocity R2: {r2_velocity:.2f}")
    
    # Add velocity magnitude columns
    df_pred = velocity_magnitude(df_pred)
    # Add prediction error
    df_pred = prediction_error(df_pred)

    # Save predictions if requested
    if output_csv:
        df_pred.to_csv(output_csv, index=False)

    # Plot results
    plot_comparison(df_pred, simfile, plot_solids=False)