"""
File: bc_train.py
Author: Eli Yale

Behavior Cloning for AcousticWorld
Loads demonstrations, trains BC policy, saves model
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))

############################################
# Dataset
############################################

class DemoDataset(Dataset):
    def __init__(self, demo_file):
        data = np.load(demo_file)
        self.obs = data["observations"].astype(np.float32)
        self.acts = data["actions"].astype(np.float32)

        print(f"Loaded demos: {self.obs.shape[0]} samples")
        print(f"Obs dim: {self.obs.shape[1]}, Act dim: {self.acts.shape[1]}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


############################################
# Behavior Cloning Network
############################################

class BCNet(nn.Module):
    def __init__(self, input_dim):
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


############################################
# Training Loop
############################################

def train_bc(demo_file="demos.npz", epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = DemoDataset(demo_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.obs.shape[1]
    model = BCNet(input_dim).to(device)

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

def load_policy(model_path="bc_policy.pt", input_dim=None):
    if input_dim is None:
        raise ValueError("You must specify input_dim")

    model = BCNet(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


############################################
# Main
############################################

if __name__ == "__main__":
    demo_file = os.path.join(script_dir, "..", "data", "demos.npz")

    train_bc(
        demo_file=demo_file,
        epochs=30,
        batch_size=256,
        lr=1e-3
    )
