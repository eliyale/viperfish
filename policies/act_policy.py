'''
File: act.py

Author: Eli Yale

Description: The action chunking transformer model implemented in pytorch
'''
"""
File: act_policy.py

Author: Eli Yale

Description:
Action Chunking Transformer (ACT) policy module for AcousticWorld.
Predicts chunks of future actions given sequences of observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------
# Hyperparameters
# --------------------
demo_file = "../data/demos/acoustic_demos_v1.npz"
obs_dim = 20        # make sure this matches your INPUT_DIM
action_dim = 2
chunk_size = 5
d_model = 128
num_layers = 2
num_heads = 4
dropout = 0.1
batch_size = 64
lr = 1e-3
num_epochs = 20
device = "cpu"  # M2 Max can train small models on CPU

class ACTPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,         # input dimension per timestep
                 action_dim: int = 2,  # number of continuous action outputs
                 d_model: int = 128,   # transformer embedding size
                 num_layers: int = 2,  # number of transformer encoder layers
                 num_heads: int = 4,   # number of attention heads
                 chunk_size: int = 5,  # number of future steps to predict at once
                 dropout: float = 0.1):
        """
        obs_dim: size of observation vector per timestep
        action_dim: size of output action vector per timestep (e.g., [steer, accel])
        chunk_size: number of actions to predict in one chunk
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Project obs_dim -> d_model
        self.obs_embedding = nn.Linear(obs_dim, d_model)

        # Positional encoding for chunk sequence
        self.pos_embedding = nn.Parameter(torch.randn(chunk_size, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        # Output head: predict actions for each timestep in chunk
        # Produces (batch, chunk_size, action_dim)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, obs_chunk: torch.Tensor):
        """
        obs_chunk: (batch, chunk_size, obs_dim)
        Returns:
            actions_pred: (batch, chunk_size, action_dim)
        """
        # Embed observations
        x = self.obs_embedding(obs_chunk)  # (batch, chunk_size, d_model)

        # Add positional encoding
        x = x + self.pos_embedding.unsqueeze(0)  # broadcast to batch

        # Pass through transformer encoder
        x = self.transformer(x)  # (batch, chunk_size, d_model)

        # Predict actions for each timestep in the chunk
        actions_pred = self.action_head(x)  # (batch, chunk_size, action_dim)
        return actions_pred


class ACTPolicyWrapper:
    """
    Wraps the ACTPolicy so it has a .act() method like RandomPolicy or BCPolicyWrapper.
    For evaluation: predict chunk, execute only first action
    """
    def __init__(self, model: ACTPolicy, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.chunk_size = model.chunk_size
        self.last_obs_chunk = None  # keep last chunk for evaluation

    def act(self, obs: torch.Tensor):
        """
        obs: np.array or torch.Tensor, shape (obs_dim,)
        Returns: np.array shape (action_dim,) -> first action in predicted chunk
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.to(self.device)

        # Initialize chunk if first step
        if self.last_obs_chunk is None:
            self.last_obs_chunk = obs.unsqueeze(0).repeat(self.chunk_size, 1)  # (chunk_size, obs_dim)
        else:
            # Shift chunk left and append new obs
            self.last_obs_chunk = torch.cat([self.last_obs_chunk[1:], obs.unsqueeze(0)], dim=0)

        obs_chunk = self.last_obs_chunk.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            actions_chunk = self.model(obs_chunk)  # (1, chunk_size, action_dim)
            action = actions_chunk[0, 0].cpu().numpy()  # take first action in chunk

        return action
