import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from collections import deque, namedtuple
import random
import torch.nn.functional as F
from itertools import count

# DQN Architecture
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)


# Atari Environment Preprocessor
class AtariPreprocessor:
    def __init__(self, new_size, num_frames=4):
        self.new_size = new_size
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def process_state_for_network(self, state):
        processed_state = Image.fromarray(state).convert('L').resize(self.new_size)
        processed_state = np.array(processed_state, dtype=np.float32) / 255.0
        self.frames.append(processed_state)
        if len(self.frames) < self.num_frames:
            # Repeat the first frame if we don't have enough frames yet
            return np.stack([self.frames[0]] * (self.num_frames - len(self.frames)) + list(self.frames), axis=0)
        return np.stack(self.frames, axis=0)
    
    def process_reward(self, reward):
        return np.clip(reward, -1, 1)
    
    def process_state_for_memory(self, state):
        return np.expand_dims(state, 0)

    def reset(self):
        self.frames.clear()

# Replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class GreedyPolicy:
    """Always returns the best action according to Q-values."""

    def select_action(self, q_values):
        """Selects the action with the highest Q-value.

        Args:
            q_values: The Q-values for all actions, as a PyTorch tensor.

        Returns:
            int: The index of the action with the highest Q-value.
        """
        return q_values.max(1)[1].view(1, 1)  # Returns the action index with the highest Q-value

# Training hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Environment setup
env = gymnasium.make('Enduro-v0', render_mode="rgb_array")
input_shape = (4, 84, 84)  # (Channels, Height, Width)
n_actions = env.action_space.n

# Initialize network and optimizer
policy_net = DQN(input_shape, n_actions).float()
target_net = DQN(input_shape, n_actions).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set the target net to evaluation mode

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayBuffer(10000)
preprocessor = AtariPreprocessor((84, 84), 4)

steps_done = 0

def select_action(state, policy_net, steps_done, evaluation=False):
    global EPS_END, EPS_START, EPS_DECAY
    if evaluation:
        with torch.no_grad():
            return greedy_policy.select_action(policy_net(state))
    else:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # Not enough samples in the memory yet

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

greedy_policy = GreedyPolicy()

# Training loop
num_episodes = 50
for episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    preprocessor.reset()  # Reset the preprocessor frame stack
    last_screen = preprocessor.process_state_for_network(env.render())
    current_screen = preprocessor.process_state_for_network(env.render())
    state = torch.tensor(current_screen - last_screen, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    for t in count():
        # Select and perform an action
        action = select_action(state, policy_net, steps_done)
        _, reward, done,_ , _ = env.step(action.item())
        reward = preprocessor.process_reward(reward)

        # Observe new state
        last_screen = current_screen
        current_screen = preprocessor.process_state_for_network(env.render())
        if not done:
            next_state = torch.tensor(current_screen - last_screen, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        if done:
            break

    # Evaluation phase at the end of each episode
    total_reward = 0
    env.reset()
    preprocessor.reset()  # Reset the preprocessor frame stack
    state = preprocessor.process_state_for_network(env.render())
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    for t in count():
        action = select_action(state, policy_net, steps_done, evaluation=True)
        state, reward, done, _ = env.step(action.item())
        state = preprocessor.process_state_for_network(env.render())
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        total_reward += reward
        if done:
            print(f"Episode {episode}: Total Reward (Evaluation) = {total_reward}")
            break

    # Update the target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
