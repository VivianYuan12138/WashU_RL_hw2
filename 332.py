from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.model import DQN
from deeprl_hw2.core import ReplayMemory, Preprocessor
from deeprl_hw2.policy import UniformRandomPolicy,GreedyEpsilonPolicy,LinearDecayGreedyEpsilonPolicy
import os
import random
import numpy as np
import torch
import gymnasium
from torch.optim.lr_scheduler import StepLR

# Hyperparameters
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 3e-4
TARGET_UPDATE_FREQ = -0.01
WEIGHT_DECAY = 1e-3

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):
    # Create a DQN model
    input_shape = (window, *input_shape)
    model = DQN(input_shape, num_actions)
    return model

def get_output_folder(parent_dir, env_name):
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

if __name__ == '__main__':
    seed = 0
    env_name = 'Enduro-v0'
    input_shape = (84, 84)
    window = 4
    output = 'atari-v0'
    output = get_output_folder(output, env_name)

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = gymnasium.make(env_name, render_mode="rgb_array")
    num_actions = env.action_space.n

    # Create Q-network
    model = create_model(window, input_shape, num_actions)
    # Create replay memory
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, window)
    preprocessor = Preprocessor(input_shape)
    policy = GreedyEpsilonPolicy(0.1)
    num_burn_in = 50000
    train_freq = 10
    num_actions = env.action_space.n
    # Create DQN agent
    agent = DQNAgent(model, preprocessor, replay_memory, policy, GAMMA,\
                    TARGET_UPDATE_FREQ, num_burn_in,train_freq, BATCH_SIZE, num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    agent.compile(optimizer, mean_huber_loss)

    # Train agent
    agent.fit(env, 10000, max_episode_length=10000)

    # Save model
    torch.save(model.state_dict(), os.path.join(output, 'model.pth'))



  
    
