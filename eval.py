'''
File: eval.py

Author: Eli Yale

Description: Evaluation of the ACT policy in the simulated acoutic world
'''
from acoustic_world import AcousticWorld
from random_policy import RandomPolicy

env = AcousticWorld(render=False)
policy = RandomPolicy()

def evaluate_policy(env, policy, max_time=60.0, dt=0.1):
    obs = env.reset()
    t = 0
    distance = 0
    collisions = 0

    while t < max_time:
        action = policy.act(obs)
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

print(evaluate_policy(env, policy))