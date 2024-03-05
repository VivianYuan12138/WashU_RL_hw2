import numpy as np
import random
from collections import deque
import gym

class DoubleQNetwork:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.W1 = np.random.rand(state_dim, action_dim)  # Weight matrix for online network
        self.W2 = np.random.rand(state_dim, action_dim)  # Weight matrix for target network
        self.learning_rate = learning_rate

    def predict(self, state):
        return np.dot(state, self.W1)

    def target_predict(self, state):
        return np.dot(state, self.W2)

    def update(self, state, action, target):
        prediction = self.predict(state)
        error = target - prediction
        self.W1 += self.learning_rate * np.outer(state, error)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

class Agent:
    def __init__(self, state_dim, action_dim, replay_buffer_capacity=10000, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq
        self.steps = 0
        self.online_network = DoubleQNetwork(state_dim, action_dim)
        self.target_network = DoubleQNetwork(state_dim, action_dim)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            q_values = self.online_network.predict(state)
            return np.argmax(q_values)

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Double Q-learning update
        online_q_values = self.online_network.predict(next_states)
        best_actions = np.argmax(online_q_values, axis=1)
        target_q_values = self.target_network.target_predict(next_states)
        targets = rewards + self.gamma * target_q_values[np.arange(self.batch_size), best_actions] * (1 - dones)

        for i, action in enumerate(actions):
            target = targets[i]
            state = states[i]
            self.online_network.update(state, action, target)

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            # Update target network weights with online network weights
            self.target_network.W1 = self.online_network.W1.copy()
            self.target_network.W2 = self.online_network.W2.copy()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# Environment settings
env = gym.make('Breakout-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Agent settings
agent = Agent(state_dim, action_dim)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward
        state = next_state
    print("Episode:", episode, "Total Reward:", total_reward)

env.close()
