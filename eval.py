"""
File: eval.py

Author: Eli Yale

Description: Evaluation of the ACT policy in the simulated acoustic world
"""

import argparse
import os
import sys
import numpy as np
import torch

# ------------------------------
# Add project root to path
# ------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acoustic_world import AcousticWorld
from config.env_config import INPUT_DIM, OUTPUT_DIM
from policies.random_policy import RandomPolicy
from policies.bc_policy import BCNet, BCPolicyWrapper, DemoDataset as BCDataset
from policies.act_policy import ACTPolicy, ACTPolicyWrapper, ChunkedDataset

# ------------------------------
# Evaluation loop
# ------------------------------
def evaluate_policy(env, policy, max_time=60.0, dt=0.1):
    obs = env.reset()
    t = 0
    distance = 0
    collisions = 0

    while t < max_time:
        action = policy.act(obs)  # always returns denormalized action if needed
        obs, collision = env.step(action)

        distance += abs(env.agent.v) * dt
        if collision:
            collisions += 1
            break

        t += dt

    return {
        "survival_time": t,
        "distance_traveled": distance,
        "collisions": collisions
    }

# ------------------------------
# Load policy helpers
# ------------------------------
def load_bc_policy(demo_file, weights_file):
    from policies.bc_policy import BCNet, DemoDataset, BCPolicyWrapper
    from config.env_config import INPUT_DIM

    dataset = DemoDataset(demo_file)   # to recover normalization stats

    model = BCNet(INPUT_DIM)
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model.eval()

    return BCPolicyWrapper(model, dataset)


def load_act_policy(demo_file, weights_file, device="cpu"):
    dataset = ChunkedDataset(demo_file)
    model = ACTPolicy(obs_dim=dataset.obs_chunks.shape[2], action_dim=2, chunk_size=dataset.chunk_size)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    # Pass dataset to wrapper so it knows normalization stats
    return ACTPolicyWrapper(model, dataset=dataset, device=device)


# ------------------------------
# Main CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    parser.add_argument("--time", type=float, default=60.0, help="Max evaluation time (seconds)")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "bc", "act"], help="Policy to evaluate")
    parser.add_argument("--demo_file", type=str, default="data/demos.npz", help="Demo dataset path for BC/ACT")
    parser.add_argument("--bc_weights", type=str, default="checkpoints/bc_policy.pt", help="BC policy weights")
    parser.add_argument("--act_weights", type=str, default="checkpoints/act_policy.pt", help="ACT policy weights")
    args = parser.parse_args()

    # Environment
    env = AcousticWorld(render=args.render)

    # Load policy
    if args.policy == "random":
        policy = RandomPolicy()
    elif args.policy == "bc":
        policy = load_bc_policy(args.demo_file, args.bc_weights)
    elif args.policy == "act":
        policy = load_act_policy(args.demo_file, args.act_weights)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Evaluate
    results = evaluate_policy(env, policy, max_time=args.time)
    print("\n=== Evaluation Results ===")
    print(f"Policy: {args.policy}")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()