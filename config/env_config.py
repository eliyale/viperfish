# configs/env_config.py

NUM_RAYS = 17         # number of sonar rays
USE_HEADING = True    # whether heading and angular velocity are included
USE_ACCEL = True      # include agent accel in input

# Derived input dimension
INPUT_DIM = NUM_RAYS
if USE_ACCEL:
    INPUT_DIM += 1
if USE_HEADING:
    INPUT_DIM += 2    # heading + angular vel

OUTPUT_DIM = 2  # steering, throttle

CLOCK_RATE = 30

WIDTH, HEIGHT = 800, 600
NUM_OBSTACLES = 10
MAX_RANGE = 200 
DT = 0.02 # 20 ms per step = 50fps
BG_COLOR = (37, 57, 69)
OBSTACLE_COLOR = (122, 13, 13)
AGENT_COLOR = (92, 12, 92)
RAY_COLOR = (0,100,255)
AGENT_RADIUS = 8
