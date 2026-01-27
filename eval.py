"""
File: eval.py

Author: Eli Yale

Description: Evaluation of the ACT policy in the simulated acoustic world
"""

import argparse
from acoustic_world import AcousticWorld
from random_policy import RandomPolicy


def evaluate_policy(env, policy, max_time=60.0, dt=0.1):
    obs = env.reset()
    t = 0
    distance = 0
    collisions = 0

    while t < max_time:
        action = policy.act(obs)
        print("Action:", action)

        obs, collision = env.step(action)

        # Track distance traveled
        distance += abs(env.agent.v) * dt

        if collision:
            collisions += 1
            break   # stop at first collision (for survival time metric)

        t += dt

    return {
        "survival_time": t,
        "distance_traveled": distance,
        "collisions": collisions
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Max evaluation time (seconds)")
    args = parser.parse_args()

    env = AcousticWorld(render=args.render)
    policy = RandomPolicy()

    results = evaluate_policy(env, policy, max_time=args.time)
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
