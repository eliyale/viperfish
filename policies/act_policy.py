"""
File: act_policy.py

Author: Eli Yale

Description:
Action Chunking Transformer (ACT) policy module for AcousticWorld.
Includes:
- ACTPolicy architecture
- ACTPolicyWrapper for evaluation
- Training loop from chunked demo dataset
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

# -------------------------------
# ACTPolicy Model
# -------------------------------
class ACTPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,         # input dimension per timestep
                 action_dim: int = 2,  # number of continuous action outputs
                 d_model: int = 128,   # transformer embedding size
                 num_layers: int = 2,  # number of transformer encoder layers
                 num_heads: int = 4,   # number of attention heads
                 chunk_size: int = 5,  # number of future steps to predict at once
                 dropout: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Linear embedding of obs
        self.obs_embedding = nn.Linear(obs_dim, d_model)

        # Positional encoding for chunk sequence
        self.pos_embedding = nn.Parameter(torch.randn(chunk_size, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        # Output head
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, obs_chunk: torch.Tensor):
        """
        obs_chunk: (batch, chunk_size, obs_dim)
        Returns:
            actions_pred: (batch, chunk_size, action_dim)
        """
        x = self.obs_embedding(obs_chunk)  # (B, chunk_size, d_model)
        x = x + self.pos_embedding.unsqueeze(0)  # broadcast to batch
        x = self.transformer(x)  # (B, chunk_size, d_model)
        actions_pred = self.action_head(x)  # (B, chunk_size, action_dim)
        return actions_pred


# -------------------------------
# Wrapper for evaluation
# -------------------------------
class ACTPolicyWrapper:
    """
    Wraps ACTPolicy to have .act() like RandomPolicy or BCPolicy
    """
    def __init__(self, model: ACTPolicy, dataset=None, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.chunk_size = model.chunk_size
        self.last_obs_chunk = None

        if dataset is not None:
            self.obs_mean = dataset.obs_mean
            self.obs_std = dataset.obs_std
            self.act_mean = dataset.act_mean
            self.act_std = dataset.act_std
        else:
            # fallback (no normalization)
            self.obs_mean = 0.0
            self.obs_std = 1.0
            self.act_mean = 0.0
            self.act_std = 1.0

    def act(self, obs: np.ndarray):
        # Normalize obs
        obs_norm = (obs - self.obs_mean) / self.obs_std
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)

        if self.last_obs_chunk is None:
            self.last_obs_chunk = obs_tensor.unsqueeze(0).repeat(self.chunk_size, 1)
        else:
            self.last_obs_chunk = torch.cat([self.last_obs_chunk[1:], obs_tensor.unsqueeze(0)], dim=0)

        obs_chunk = self.last_obs_chunk.unsqueeze(0)  # batch dim
        with torch.no_grad():
            actions_chunk_norm = self.model(obs_chunk)  # (1, chunk_size, action_dim)
            action_norm = actions_chunk_norm[0, 0].cpu().numpy()

        # Denormalize action
        action = action_norm * self.act_std + self.act_mean
        return action


class ChunkedDataset(Dataset):
    """
    Converts demo dataset to overlapping chunks for ACT training
    and normalizes observations and actions.
    """
    def __init__(self, demo_file: str, chunk_size: int = 5):
        # Load raw demo data
        data = np.load(demo_file)
        obs = data['observations']   # (N, obs_dim)
        actions = data['actions']    # (N, action_dim)

        # -------------------------------
        # Compute normalization stats
        # -------------------------------
        self.obs_mean = obs.mean(axis=0)
        self.obs_std = obs.std(axis=0) + 1e-8  # prevent division by zero
        self.act_mean = actions.mean(axis=0)
        self.act_std = actions.std(axis=0) + 1e-8

        # Normalize
        obs = (obs - self.obs_mean) / self.obs_std
        actions = (actions - self.act_mean) / self.act_std

        # -------------------------------
        # Create overlapping chunks
        # -------------------------------
        self.chunk_size = chunk_size
        self.obs_chunks = []
        self.action_chunks = []

        N = len(obs)
        for i in range(N - chunk_size):
            self.obs_chunks.append(obs[i:i+chunk_size])
            self.action_chunks.append(actions[i:i+chunk_size])

        # Convert to tensors
        self.obs_chunks = torch.tensor(np.array(self.obs_chunks), dtype=torch.float32)
        self.action_chunks = torch.tensor(np.array(self.action_chunks), dtype=torch.float32)

    def __len__(self):
        return len(self.obs_chunks)

    def __getitem__(self, idx):
        return self.obs_chunks[idx], self.action_chunks[idx]

    # Optional: helper methods to denormalize predictions
    def denormalize_actions(self, actions):
        """
        actions: torch.Tensor or np.array
        """
        return actions * self.act_std + self.act_mean

    def denormalize_obs(self, obs):
        return obs * self.obs_std + self.obs_mean

# -------------------------------
# Training function
# -------------------------------
def train_act(demo_file: str,
              obs_dim: int,
              action_dim: int = 2,
              chunk_size: int = 5,
              d_model: int = 128,
              num_layers: int = 2,
              num_heads: int = 4,
              dropout: float = 0.1,
              batch_size: int = 64,
              lr: float = 1e-3,
              num_epochs: int = 20,
              device: str = "cpu",
              save_path: str = "../checkpoints/act_policy.pt"):
    # Dataset
    dataset = ChunkedDataset(demo_file, chunk_size=chunk_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = ACTPolicy(obs_dim, action_dim, d_model, num_layers, num_heads, chunk_size, dropout)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for obs_chunk, action_chunk in loader:
            obs_chunk = obs_chunk.to(device)
            action_chunk = action_chunk.to(device)

            optimizer.zero_grad()
            pred_chunk = model(obs_chunk)
            loss = criterion(pred_chunk, action_chunk)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * obs_chunk.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.6f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"ACT model saved to {save_path}")
    return model


# -------------------------------
# Main function for CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, default="../data/demos.npz")
    parser.add_argument("--obs_dim", type=int, default=20)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="../checkpoints/act_policy.pt")
    args = parser.parse_args()

    train_act(demo_file=args.demo_file,
              obs_dim=args.obs_dim,
              action_dim=args.action_dim,
              chunk_size=args.chunk_size,
              d_model=args.d_model,
              num_layers=args.num_layers,
              num_heads=args.num_heads,
              dropout=args.dropout,
              batch_size=args.batch_size,
              lr=args.lr,
              num_epochs=args.num_epochs,
              device=args.device,
              save_path=args.save_path)


if __name__ == "__main__":
    main()
