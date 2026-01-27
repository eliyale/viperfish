import numpy as np

import numpy as np

class RandomPolicy:
    def __init__(self, steer_scale=1.0, target_speed=20, change_prob=0.05):
        self.steer_scale = steer_scale
        self.target_speed = target_speed
        self.change_prob = change_prob
        self.current_steer = 0.0

    def act(self, obs):
        # obs[-1] assumed normalized velocity
        v = obs[-1] * 200  # unnormalize if you normalized v in env

        # Randomly change steering direction
        if np.random.rand() < self.change_prob:
            self.current_steer = np.random.uniform(-self.steer_scale, self.steer_scale)

        steer = self.current_steer

        # Simple cruise controller
        if v < self.target_speed:
            accel = 20
        else:
            accel = 0

        return np.array([steer, accel])

