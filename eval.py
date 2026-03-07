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
import pickle

# ------------------------------
# Add project root to path
# ------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acoustic_world import AcousticWorld
from config.env_config import DT, INPUT_DIM, OUTPUT_DIM
from policies.random_policy import RandomPolicy
from policies.bc_policy import BCNet, BCPolicyWrapper, DemoDataset as BCDataset
from policies.act_policy import ACTPolicy, ACTPolicyWrapper, ChunkedDataset

# ------------------------------
# Evaluation loop
# ------------------------------
def evaluate_policy(env, policy, max_time=60.0):
    obs = env.reset()
    t = 0
    distance = 0
    collisions = 0

    while t < max_time:
        action = policy.act(obs)
        obs, collision = env.step(action)

        if env.render_mode:
            env.render()

        # Track stats
        distance += abs(env.agent.v) * DT
        if collision:
            collisions += 1
            obs = env.reset() 

        t += DT

    return {
        "survival_time": t,
        "distance_traveled": distance,
        "total_collisions": collisions,
        "success": 1 if collisions == 0 else 0,
        "collisions_per_minute": collisions / (max_time / 60.0)
    }

# ------------------------------
# Load policy helpers
# ------------------------------
def load_bc_policy(weights_file, stats_file):
    # 1. Load the pickled stats
    with open(stats_file, "rb") as f:
        stats = pickle.load(f)

    # 2. Load the model
    from policies.bc_policy import BCNet, BCPolicyWrapper
    model = BCNet(INPUT_DIM)
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model.eval()

    # 3. Wrap it with just the stats
    return BCPolicyWrapper(model, stats)


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
    parser.add_argument("--bc_weights", type=str, default="checkpoints/bc_policy.pt", help="BC policy weights")
    parser.add_argument("--act_weights", type=str, default="checkpoints/act_policy.pt", help="ACT policy weights")
    args = parser.parse_args()

    # Environment
    env = AcousticWorld(render=args.render)

    # Load policy
    if args.policy == "random":
        policy = RandomPolicy()
    elif args.policy == "bc":
        policy = load_bc_policy(args.bc_weights, "checkpoints/bc_stats.pkl")
    elif args.policy == "act":
        raise NotImplementedError()
        policy = load_act_policy(args.demo_file, args.act_weights)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Evaluate
    num_episodes = 10
    all_results = []

    print(f"Evaluating {args.policy} for {num_episodes} episodes...")

    for i in range(num_episodes):
        res = evaluate_policy(env, policy, max_time=args.time)
        all_results.append(res)
        print(f"Episode {i+1}: {res['total_collisions']} collisions")

    # Aggregate Results
    avg_collisions = np.mean([r['total_collisions'] for r in all_results])
    success_rate = np.mean([r['success'] for r in all_results]) * 100
    avg_dist = np.mean([r['distance_traveled'] for r in all_results])

    print("\n" + "="*30)
    print(f"FINAL EVALUATION: {args.policy}")
    print("="*30)
    print(f"Success Rate (0 collisions): {success_rate:.1f}%")
    print(f"Avg Collisions per Episode: {avg_collisions:.2f}")
    print(f"Avg Distance Traveled:      {avg_dist:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()