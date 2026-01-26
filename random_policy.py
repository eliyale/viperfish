import numpy as np

class RandomPolicy:
    def __init__(self, steer_scale=1.0, accel=0.0, change_prob=0.1):
        """
        steer_scale: max steering rate
        accel: fixed acceleration (0 means constant velocity)
        change_prob: probability of changing steering direction each step
        """
        self.steer_scale = steer_scale
        self.accel = accel
        self.change_prob = change_prob
        self.current_steer = 0.0

    def act(self, obs):
        # Occasionally change steering direction
        if np.random.rand() < self.change_prob:
            self.current_steer = np.random.uniform(-self.steer_scale, self.steer_scale)

        steer = self.current_steer
        accel = self.accel
        return np.array([steer, accel])
