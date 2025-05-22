import os
import re
import time
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from joblib import Parallel, delayed
from datetime import datetime
from preprocessing import FluidGridInterpolator


# --------------- Model Definition ---------------
class FanBladeNN(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3)
        )
        self.pressure_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        pressure = self.pressure_head(encoded)
        velocity = self.velocity_head(encoded)
        return pressure, velocity




# --------------- Helper Functions ---------------
def extract_parameters_from_filename(filename):
    """Extract input parameters from the files on the form AA_BB_CC.csv"""
    match = re.search(r'(\d+)_([\d.]+)_(\d+)', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match pattern.")
    return [float(match.group(1)), float(match.group(2)), float(match.group(3))]


def chunk_file_list(file_list, chunk_size=4):
    """Creates list of files included in chunk"""
    for i in range(0, len(file_list), chunk_size):
        yield file_list[i:i + chunk_size]


def fit_scaler_on_file(file, all_nodes_file):
    """Helper function for fitting scalers in parallel"""
    params = extract_parameters_from_filename(os.path.basename(file))
    processor = FluidGridInterpolator(file, all_nodes_file)
    df = processor.process()
    return params, df


def fit_scalers_on_files(train_files, all_nodes_file, n_jobs=4):
    """Fit the scalers on all files when all files are processed together"""
    print(f"Using {n_jobs} CPU cores for parallel scaler fitting.")
    param_scaler = StandardScaler()
    spatial_scaler = StandardScaler()
    pressure_scaler = StandardScaler()
    velocity_scaler = StandardScaler()
    # Parallel processing of files
    results = Parallel(n_jobs=n_jobs)(delayed(fit_scaler_on_file)(file, all_nodes_file) for file in train_files)
    # Incremental fitting
    for params, df in results:
        param_scaler.partial_fit(np.array([params]).astype(np.float32))
        spatial_scaler.partial_fit(df[['X (m)', 'Y (m)', 'Z (m)']].values)
        pressure_scaler.partial_fit(df['Pressure (Pa)'].values.reshape(-1, 1))
        velocity_scaler.partial_fit(df[['Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)']].values)

    return param_scaler, spatial_scaler, pressure_scaler, velocity_scaler


def process_single_file(file, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler):
    """Process individual files and scale using fitted scalers"""
    params = extract_parameters_from_filename(os.path.basename(file))
    params_scaled = param_scaler.transform(np.array([params]).astype(np.float32)).flatten()
    processor = FluidGridInterpolator(file, all_nodes_file)
    df = processor.process()
    num_nodes = len(df)
    X = np.zeros((num_nodes, 7), dtype=np.float32)
    Y = np.zeros((num_nodes, 4), dtype=np.float32)
    X[:, 0:3] = params_scaled
    coords = df[['X (m)', 'Y (m)', 'Z (m)']].values
    X[:, 3:6] = spatial_scaler.transform(coords)
    X[:, 6] = df['is_fluid'].values
    Y[:, 0] = pressure_scaler.transform(df['Pressure (Pa)'].values.reshape(-1, 1)).flatten()
    Y[:, 1:4] = velocity_scaler.transform(df[['Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)']].values)
    mask = df['is_fluid'].values.astype(np.float32)
    return X, Y, mask


def load_all_data_into_memory(file_list, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler, n_jobs=4):
    """Stack scaled data and load into memory in parallell"""
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(file, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler) for file in file_list
    )
    X_all = np.vstack([r[0] for r in results])
    Y_all = np.vstack([r[1] for r in results])
    mask_all = np.hstack([r[2] for r in results])
    return X_all, Y_all, mask_all




# --------------- Loss Function ---------------
def mse_loss(pred_pressure, pred_velocity, target_pressure, target_velocity, mask, pressure_weight=1.0, velocity_weight=1.0):
    """Compute Mean Squared Error on all nodes, both fluid and solid nodes"""
    pressure_loss = (pred_pressure - target_pressure) ** 2
    velocity_loss = (pred_velocity - target_velocity) ** 2
    mean_pressure_loss = pressure_loss.mean()
    mean_velocity_loss = velocity_loss.mean()
    total_loss = pressure_weight * mean_pressure_loss + velocity_weight * mean_velocity_loss
    return total_loss, mean_pressure_loss, mean_velocity_loss



# --------------- Saving and loading checkpoints ---------------
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, log, train_loss_history, val_loss_history, checkpoint_path):
    """Save epoch checkpoints"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'log': log,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint from last epoch"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    log = checkpoint['log']
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    return model, optimizer, scheduler, epoch, best_val_loss, log, train_loss_history, val_loss_history




# --------------- Training/Validation Chunks ---------------
def train_on_chunk(model, optimizer, X_chunk, Y_chunk, mask_chunk, batch_size, loss_fn, device):
    """Training loop"""
    model.train()
    X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_chunk, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_chunk, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    total_loss = 0.0
    total_p_loss = 0.0
    total_v_loss = 0.0
    total_batches = 0
    optimizer.zero_grad()
    for i, (X_batch, Y_batch, mask_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        Y_pressure = Y_batch[:, 0:1]
        Y_velocity = Y_batch[:, 1:4]
        pressure_pred, velocity_pred = model(X_batch)
        loss, p_loss, v_loss = loss_fn(pressure_pred, velocity_pred, Y_pressure, Y_velocity, mask_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
        total_batches += 1
    avg_loss = total_loss / total_batches
    avg_p_loss = total_p_loss / total_batches
    avg_v_loss = total_v_loss / total_batches
    return avg_loss, avg_p_loss, avg_v_loss

def validate_on_chunk(model, X_chunk, Y_chunk, mask_chunk, batch_size, loss_fn, device):
    """Validation loop"""
    model.eval()
    X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_chunk, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_chunk, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    total_p_loss = 0.0
    total_v_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for X_batch, Y_batch, mask_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            mask_batch = mask_batch.to(device)
            Y_pressure = Y_batch[:, 0:1]
            Y_velocity = Y_batch[:, 1:4]
            pressure_pred, velocity_pred = model(X_batch)
            loss, p_loss, v_loss = mse_loss(pressure_pred, velocity_pred, Y_pressure, Y_velocity, mask_batch)
            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            total_batches += 1
    avg_loss = total_loss / total_batches
    avg_p_loss = total_p_loss / total_batches
    avg_v_loss = total_v_loss / total_batches
    return avg_loss, avg_p_loss, avg_v_loss




# --------------- Main Training Function ---------------
def train_model(
    data_dir,
    all_nodes_file,
    output_dir='output_data',
    last_checkpoint_path='last_checkpoint.pt',
    best_checkpoint_path= 'best_checkpoint.pt',
    initial_checkpoint_path=None,
    scalers_path=None,
    log_path='log.json',
    batch_size=1024,
    chunk_size=4,
    num_epochs=100,
    validation_split=0.2,
    learning_rate=1e-3,
    pressure_weight=1.0,
    velocity_weight=1.0,
    continue_training=False,
    fit_scalers=True,
    load_in_chunks=True,
    n_jobs=8
):
    """ Main training loop, preprocessing, scaling, loading, training, validation etc."""
    torch.set_num_threads(n_jobs)

    os.makedirs(output_dir, exist_ok=True)
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_files.sort()
    train_files, val_files = train_test_split(all_files, test_size=validation_split, random_state=42)
    print(f"Found {len(all_files)} simulation files")
    print(f"{len(train_files)} training files")
    #print(f"{[os.path.basename(f) for f in train_files]}")
    print(f"{len(val_files)} validation files")
    #print(f"{[os.path.basename(f) for f in val_files]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    scalers_file = scalers_path
    if fit_scalers:
        print("Fitting scalers on all training files (full memory)...")
        start_time = time.time()
        param_scaler, spatial_scaler, pressure_scaler, velocity_scaler = fit_scalers_on_files(train_files, all_nodes_file, n_jobs)
        end_time = time.time()
        print(f"Scaler fitting took {end_time - start_time:.2f} seconds")
        scalers = {
            'param_scaler': param_scaler,
            'spatial_scaler': spatial_scaler,
            'pressure_scaler': pressure_scaler,
            'velocity_scaler': velocity_scaler
        }
        joblib.dump(scalers, scalers_file)
        print(f"Scalers saved to {scalers_file}")
    else:
        if os.path.exists(scalers_file):
            print(f"Loading scalers from {scalers_file}")
            scalers = joblib.load(scalers_file)
            param_scaler = scalers['param_scaler']
            spatial_scaler = scalers['spatial_scaler']
            pressure_scaler = scalers['pressure_scaler']
            velocity_scaler = scalers['velocity_scaler']
        else:
            raise RuntimeError(f"Scalers file not found at {scalers_file}. Please fit and save scalers first.")


    num_inputs = 7  # 3 params + 3 coords + 1 is_fluid
    model = FanBladeNN(num_inputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 0
    if continue_training and os.path.exists(initial_checkpoint_path):
        print(f"Loading model from {initial_checkpoint_path} and resumes training")
        model, optimizer, scheduler, start_epoch, best_val_loss, log, train_loss_history, val_loss_history = load_checkpoint(
            model, optimizer, scheduler, initial_checkpoint_path, device
        )
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    else:
        print("Starting training from scratch")
        best_val_loss = float('inf')
        log = {
            "date": datetime.now().isoformat(),
            "data_dir": data_dir,
            "all_nodes_file": all_nodes_file,
            "train_files": [os.path.basename(f) for f in train_files],
            "val_files": [os.path.basename(f) for f in val_files],
            "hyperparameters": {
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "pressure_weight": pressure_weight,
                "velocity_weight": velocity_weight,
                "validation_split": validation_split,
                "load_in_batches": load_in_chunks,
                "n_jobs": n_jobs
            },
            "random_seed": 42,
            "model_architecture": str(model),
            "device": str(device),
            "device_0": str(next(model.parameters()).device),
            "scalers_path": scalers_file,
            "training": {
                "epoch_times": [],
                "train_loss_history": [],
                "val_loss_history": [],
                "train_pressure_loss": [],
                "train_velocity_loss": [],
                "val_pressure_loss": [],
                "val_velocity_loss": [],
                "lr_history": []
            },
            "best_model": {
                "epoch": None,
                "val_loss": None,
                "train_loss": None,
                "model_path": os.path.join(output_dir, 'best_checkpoint.pt')
            },
            "software_versions": {
                "python": sys.version,
                "torch": torch.__version__
            }
        }
        train_loss_history = []
        val_loss_history = []


    if not load_in_chunks:
        # ------ Using all files ------
        print("Loading all training and validation data into memory")
        start_time = time.time()
        X_train, Y_train, mask_train = load_all_data_into_memory(
            train_files, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler, n_jobs=n_jobs
        )
        X_val, Y_val, mask_val = load_all_data_into_memory(
            val_files, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler, n_jobs=n_jobs
        )
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32),
            torch.tensor(mask_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32),
            torch.tensor(mask_val, dtype=torch.float32)
        )
        pin_memory = (device.type == 'cuda')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
        end_time = time.time()
        print(f"Loading all training and validation data into memory finished, {end_time - start_time:.2f} seconds")
    else:
        # ------ Chunked loading ------
        print("Using chunked loading")


    # ------ Training Loop ------
    print("Starting training")
    for epoch in range(start_epoch, num_epochs):

        epoch_start_time = time.time()
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)

        if not load_in_chunks:
            # ------ Using all files ------
            # Training
            model.train()
            train_losses, train_p_losses, train_v_losses = [], [], []
            for X_batch, Y_batch, mask_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)
                mask_batch = mask_batch.to(device, non_blocking=True)
                Y_pressure = Y_batch[:, 0:1]
                Y_velocity = Y_batch[:, 1:4]
                pressure_pred, velocity_pred = model(X_batch)
                loss, p_loss, v_loss = mse_loss(pressure_pred, velocity_pred, Y_pressure, Y_velocity, mask_batch, pressure_weight, velocity_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_p_losses.append(p_loss.item())
                train_v_losses.append(v_loss.item())
            epoch_train_loss = np.mean(train_losses)
            epoch_train_p_loss = np.mean(train_p_losses)
            epoch_train_v_loss = np.mean(train_v_losses)

            # Validation
            model.eval()
            val_losses, val_p_losses, val_v_losses = [], [], []
            with torch.no_grad():
                for X_batch, Y_batch, mask_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    Y_batch = Y_batch.to(device, non_blocking=True)
                    mask_batch = mask_batch.to(device, non_blocking=True)
                    Y_pressure = Y_batch[:, 0:1]
                    Y_velocity = Y_batch[:, 1:4]
                    pressure_pred, velocity_pred = model(X_batch)
                    loss, p_loss, v_loss = mse_loss(pressure_pred, velocity_pred, Y_pressure, Y_velocity, mask_batch, pressure_weight, velocity_weight)
                    val_losses.append(loss.item())
                    val_p_losses.append(p_loss.item())
                    val_v_losses.append(v_loss.item())
            epoch_val_loss = np.mean(val_losses)
            epoch_val_p_loss = np.mean(val_p_losses)
            epoch_val_v_loss = np.mean(val_v_losses)
        else:
            # ------ Chunked loading ------
            # Training
            model.train()
            train_losses, train_p_losses, train_v_losses = [], [], []
            for file_chunk in chunk_file_list(train_files, chunk_size):
                X_chunk, Y_chunk, mask_chunk = load_all_data_into_memory(
                    file_chunk, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler, n_jobs=1
                )
                avg_loss, avg_p_loss, avg_v_loss = train_on_chunk(
                    model, optimizer, X_chunk, Y_chunk, mask_chunk, batch_size, mse_loss, device)
                train_losses.append(avg_loss)
                train_p_losses.append(avg_p_loss)
                train_v_losses.append(avg_v_loss)
            epoch_train_loss = np.mean(train_losses)
            epoch_train_p_loss = np.mean(train_p_losses)
            epoch_train_v_loss = np.mean(train_v_losses)

            # Validation
            model.eval()
            val_losses, val_p_losses, val_v_losses = [], [], []
            for file_chunk in chunk_file_list(val_files, chunk_size):
                X_chunk, Y_chunk, mask_chunk = load_all_data_into_memory(
                    file_chunk, all_nodes_file, param_scaler, spatial_scaler, pressure_scaler, velocity_scaler, n_jobs=1
                )
                avg_loss, avg_p_loss, avg_v_loss = validate_on_chunk(
                    model, X_chunk, Y_chunk, mask_chunk, batch_size, mse_loss, device
                )
                val_losses.append(avg_loss)
                val_p_losses.append(avg_p_loss)
                val_v_losses.append(avg_v_loss)
            epoch_val_loss = np.mean(val_losses)
            epoch_val_p_loss = np.mean(val_p_losses)
            epoch_val_v_loss = np.mean(val_v_losses)

        epoch_time = time.time() - epoch_start_time
        scheduler.step(epoch_val_loss)


        # ------ Logging & checkpointing ------
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        log["training"]["epoch_times"].append(epoch_time)
        log["training"]["train_loss_history"].append(epoch_train_loss)
        log["training"]["val_loss_history"].append(epoch_val_loss)
        log["training"]["train_pressure_loss"].append(epoch_train_p_loss)
        log["training"]["train_velocity_loss"].append(epoch_train_v_loss)
        log["training"]["val_pressure_loss"].append(epoch_val_p_loss)
        log["training"]["val_velocity_loss"].append(epoch_val_v_loss)
        log["training"]["lr_history"].append(optimizer.param_groups[0]['lr'])

        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

        print(f"\nEpoch {epoch+1} complete in {epoch_time:.2f}s")
        print(f"  Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        print(f"  Train Pressure Loss: {epoch_train_p_loss:.6f}, Val Pressure Loss: {epoch_val_p_loss:.6f}")
        print(f"  Train Velocity Loss: {epoch_train_v_loss:.6f}, Val Velocity Loss: {epoch_val_v_loss:.6f}")

        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_val_loss, log,
            train_loss_history, val_loss_history, last_checkpoint_path
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_val_loss, log,
            train_loss_history, val_loss_history, best_checkpoint_path
            )
            log["best_model"]["epoch"] = epoch+1
            log["best_model"]["val_loss"] = float(epoch_val_loss)
            log["best_model"]["train_loss"] = float(epoch_train_loss)

            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)

    print("\nTraining complete")
    return model, scalers



# --------------- Entry Point ---------------
if __name__ == "__main__":
    config = {
        'data_dir': 'simfiles_all',                                         # Directory to simulation files
        'all_nodes_file': 'all_nodes.csv',                              # Path to file including all nodes in domain
        'output_dir': 'checkpoints',                                 # Directory for results
        'last_checkpoint_path': 'checkpoints/last_checkpoint.pt',    # Last checkpoint
        'best_checkpoint_path': 'checkpoints/best_checkpoint.pt',    # Best checkpoint
        'initial_checkpoint_path': 'checkpoints/best_checkpoint.pt',      # Path to model (either best or last) if continue_training=True
        'scalers_path': 'scalers.pkl',                   # Path to scalers
        'log_path': 'checkpoints/log.json',                         # Path to logfile
        'batch_size': 128,                                 # Number of samples in each batch
        'chunk_size': 20,                                   # Number of file in each chunk if load_in_chunks=True
        'num_epochs': 100,                                  # Number of epochs
        'validation_split': 0.2,                            # Training vs validation split
        'learning_rate': 0.001,                             # Initial learning rate
        'pressure_weight': 1.0,                             # Weighing for loss function
        'velocity_weight': 1.0,                             # Weighing for loss function
        'continue_training': False,                          # Resume training from initial_checkpoint_path
        'fit_scalers': False,                               # Fit new scalers or use existing scalers
        'load_in_chunks': False,                            # Process all files at once (False) or in chunks (True)
        'n_jobs': 16                                         # Number of parallel jobs for loading/scaling
    }

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # Train the model using the config
    model, scalers = train_model(**config)
