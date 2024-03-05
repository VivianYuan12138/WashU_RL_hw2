import torch
import torch.nn as nn
import torch.optim as optim
from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.preprocessors import AtariPreprocessor
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy

# Hyperparameters
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 3e-4
TARGET_UPDATE_FREQ = 10000  # in steps
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000

# Q-Network
class LinearQNetwork(nn.Module):
    # Define your linear model
    pass

# Instantiate components
q_network = LinearQNetwork(...)
target_network = LinearQNetwork(...)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)
preprocessor = AtariPreprocessor()
policy = LinearDecayGreedyEpsilonPolicy(...)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
agent = DQNAgent(q_network, preprocessor, memory, policy, GAMMA,
                 TARGET_UPDATE_FREQ, BATCH_SIZE)

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    state = preprocessor.process_state_for_memory(state)
    for t in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocessor.process_state_for_memory(next_state)
        memory.append(state, action, reward, next_state, done)
        state = next_state
        
        # Update Q-Network
        if memory.size() >= BATCH_SIZE:
            experiences = memory.sample(BATCH_SIZE)
            # Calculate loss and update network
            agent.update_policy(experiences)
        
        # Update Target Network
        if t % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        if done:
            break
