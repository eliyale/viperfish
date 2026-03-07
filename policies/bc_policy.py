"""
File: bc_train.py
Author: Eli Yale

Behavior Cloning for AcousticWorld
Loads demonstrations, trains BC policy, saves model
"""


import os
import sys
import glob
import pickle

# Add project root (one level up from this file) to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config.env_config import INPUT_DIM, OUTPUT_DIM


script_dir = os.path.dirname(os.path.abspath(__file__))

############################################
# Dataset
############################################

import glob

class DemoDataset(Dataset):
    def __init__(self, data_dir):
        # 1. Find all .npz files in the directory
        file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        if not file_paths:
            raise FileNotFoundError(f"No demonstration files found in {data_dir}")
        
        all_obs = []
        all_acts = []

        # 2. Iterate and load each file
        for path in file_paths:
            data = np.load(path)
            all_obs.append(data["observations"])
            all_acts.append(data["actions"])

        # 3. Concatenate into large arrays
        obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        acts = np.concatenate(all_acts, axis=0).astype(np.float32)

        # 4. Compute global normalization stats
        self.obs_mean = obs.mean(axis=0)
        self.obs_std = obs.std(axis=0) + 1e-8
        self.acts_mean = acts.mean(axis=0)
        self.acts_std = acts.std(axis=0) + 1e-8

        # 5. Normalize
        self.obs = (obs - self.obs_mean) / self.obs_std
        self.acts = (acts - self.acts_mean) / self.acts_std

        print(f"Loaded {len(file_paths)} files | Total samples: {self.obs.shape[0]}")
        print(f"Obs dim: {self.obs.shape[1]}, Act dim: {self.acts.shape[1]}")
        print("Observations and actions normalized")

        self.stats = {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "acts_mean": self.acts_mean,
            "acts_std": self.acts_std
        }

    def save_stats(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"Normalization stats saved to {path}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

    # -------------------------------
    # Helper methods to denormalize
    # -------------------------------
    def denormalize_actions(self, actions):
        """
        actions: np.array or torch.Tensor
        """
        return actions * self.acts_std + self.acts_mean

    def denormalize_obs(self, obs):
        return obs * self.obs_std + self.obs_mean


############################################
# Behavior Cloning Network
############################################

class BCNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)   # [turn, throttle]
        )

    def forward(self, x):
        return self.net(x)
    
class BCPolicyWrapper:
    """
    Wraps BCNet with normalization + act()
    """
    def __init__(self, model, stats):
        self.model = model
        self.model.eval()

        # Save normalization stats
        # Load from the stats dictionary instead of dataset object
        self.obs_mean = stats["obs_mean"]
        self.obs_std  = stats["obs_std"]
        self.act_mean = stats["acts_mean"]
        self.act_std  = stats["acts_std"]

    def act(self, obs):
        """
        obs: np.array (obs_dim,)
        returns: np.array (2,)
        """
        # Normalize observation
        obs_norm = (obs - self.obs_mean) / self.obs_std

        obs_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_norm = self.model(obs_t).numpy()[0]

        # Denormalize action
        action = action_norm * self.act_std + self.act_mean
        return action



############################################
# Training Loop
############################################

def train_bc(data_dir="data", epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = DemoDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    stats_path = os.path.join(script_dir, "..", "checkpoints", "bc_stats.pkl")
    dataset.save_stats(stats_path)

    model = BCNet(INPUT_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for obs, acts in loader:
            obs = obs.to(device)
            acts = acts.to(device)

            pred = model(obs)
            loss = loss_fn(pred, acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    save_file = os.path.join(script_dir, "..", "checkpoints", "bc_policy.pt")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    torch.save(model.state_dict(), save_file)
    print("Saved model to bc_policy.pt")


############################################
# Loading Policy (for evaluation)
############################################

def load_policy(model_path="bc_policy.pt", input_dim=INPUT_DIM):
    model = BCNet(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


############################################
# Main
############################################

if __name__ == "__main__":
    train_bc(
        data_dir="data",
        epochs=30,
        batch_size=256,
        lr=1e-3
    )
