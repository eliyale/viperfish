import argparse
import pygame
from acoustic_world import AcousticWorld
import numpy as np
import time
import os
from config.env_config import CLOCK_RATE
import datetime  # Add this import at the top

def run_environment(demos=False, out_file="demos.npz"):
    env = AcousticWorld(render=True)
    obs = env.reset()

    pygame.init()
    running = True

    # Buffers for demonstrations
    obs_buffer = []
    act_buffer = []

    print("Controls: Arrow keys to drive. ESC to quit.")

    while running:

        steer, accel = 0, 0

        # Quit window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steer = -10
        if keys[pygame.K_RIGHT]:
            steer = 10
        if keys[pygame.K_UP]:
            accel = 300
        if keys[pygame.K_DOWN]:
            accel = -300
        if keys[pygame.K_ESCAPE]:
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
    # Change the default to None so we can detect if a custom name was provided
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output demo filename (auto-generated if omitted)")
    args = parser.parse_args()

    # Generate a unique filename if collecting demos
    final_filename = args.outfile
    if args.collect_demos and final_filename is None:
        # Format: demo_20260306_132045.npz
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"demo_{timestamp}.npz"
    elif final_filename is None:
        final_filename = "demos.npz"

    run_environment(demos=args.collect_demos, out_file=final_filename)


if __name__ == "__main__":
    main()
