import numpy as np
import random
from collections import deque
import gym
from gym.wrappers import Monitor

# Define your classes and functions here...

# Environment settings
env = gym.make('Breakout-v4')  # Corrected environment name
env = Monitor(env, './videos', force=True)

# Rest of the code...
