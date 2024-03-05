import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from collections import deque, namedtuple
import random
from itertools import count
from gymnasium.wrappers import RecordVideo

# Your existing DQN, AtariPreprocessor, ReplayBuffer, and Transition classes remain unchanged

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

    def reset(self):
        self.frames.clear()

# Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# Main Training Loop Setup with Monitor
env = gymnasium.make('Enduro-v0', render_mode="rgb_array")
# env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda episode_id: episode_id % 10 == 0)
input_shape = (4, 84, 84)  # (Channels, Height, Width)
n_actions = env.action_space.n

policy_net = DQN(input_shape, n_actions).float()
target_net = DQN(input_shape, n_actions).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Evaluation mode

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayBuffer(10000)
preprocessor = AtariPreprocessor((84, 84), 4)


# Epsilon-Greedy Action Selection
def select_action(state, steps_done, eps_start=0.9, eps_end=0.05, eps_decay=200, evaluation=False):
    if evaluation:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
        if random.random() > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

# Optimization Step
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # Not enough samples in memory

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Training Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 10
num_episodes = 50
steps_done = 0

def evaluate_model(env, policy_net, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        env.reset()
        preprocessor.reset()  # Reset the preprocessor frame stack
        state = preprocessor.process_state_for_network(env.render())
        state = torch.tensor([state], dtype=torch.float32)  # Add batch dimension
        episode_reward = 0

        for t in count():
            action = select_action(state, steps_done, evaluation=True)
            _, reward, done, _, _ = env.step(action.item())
            episode_reward += reward

            if not done:
                next_state = preprocessor.process_state_for_network(env.render())
                next_state = torch.tensor([next_state], dtype=torch.float32)
                state = next_state
            else:
                break

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f'Evaluation over {num_episodes} episodes: Mean Reward = {mean_reward}, Std Reward = {std_reward}')
    return mean_reward, std_reward

# Modify the training loop to include evaluation and use the updated preprocessor
episode_rewards = []
for episode in range(num_episodes):
    env.reset()
    preprocessor.reset()  # Reset the preprocessor frame stack
    state = preprocessor.process_state_for_network(env.render())
    state = torch.tensor([state], dtype=torch.float32)  # Add batch dimension
    episode_reward = 0

    for t in count():
        action = select_action(state, steps_done)
        _, reward, done, _, _ = env.step(action.item())
        episode_reward += reward

        # Your existing code for handling transitions and optimization remains unchanged

        if done:
            episode_rewards.append(episode_reward)
            break

    # Evaluation phase
    if episode % 10 == 0 and episode > 0:
        evaluate_model(env, policy_net)

    # Update Target Network remains unchanged

env.close()
print('Training Complete')

# After training, print out the mean and std of the rewards
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(f'Overall Training Performance: Mean Reward = {mean_reward}, Std Reward = {std_reward}')
