import argparse
import pygame
from acoustic_world import AcousticWorld
import numpy as np
import time
import os
from config.env_config import CLOCK_RATE

def run_environment(demos=False, out_file="demos.npz"):
    env = AcousticWorld(render=True)
    obs = env.reset()

    pygame.init()
    clock = pygame.time.Clock()
    running = True

    # Buffers for demonstrations
    obs_buffer = []
    act_buffer = []

    print("Controls: Arrow keys to drive. ESC to quit.")

    while running:
        clock.tick(CLOCK_RATE)

        steer, accel = 0, 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steer = -2
        if keys[pygame.K_RIGHT]:
            steer = 2
        if keys[pygame.K_UP]:
            accel = 50
        if keys[pygame.K_DOWN]:
            accel = -50
        if keys[pygame.K_ESCAPE]:
            running = False

        # Quit window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.array([steer, accel], dtype=np.float32)

        # Log demo
        if demos:
            obs_buffer.append(obs.copy())
            act_buffer.append(action.copy())

        obs, collision = env.step(action)

        if collision:
            print("Collision! Resetting.")
            obs = env.reset()

    # Save demos
    if demos and len(obs_buffer) > 0:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", out_file)
        np.savez_compressed(save_path,
                            observations=np.array(obs_buffer),
                            actions=np.array(act_buffer))
        print(f"Saved {len(obs_buffer)} steps to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_demos", action="store_true",
                        help="Collect demonstrations and save to file")
    parser.add_argument("--outfile", type=str, default="demos.npz",
                        help="Output demo filename")
    args = parser.parse_args()

    run_environment(demos=args.collect_demos, out_file=args.outfile)


if __name__ == "__main__":
    main()
