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
LEARNING_RATE = 3e-5
TARGET_UPDATE_FREQ = -0.1
WEIGHT_DECAY = 1e-4


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    # Create environment
    env = gymnasium.make(env_name, render_mode="rgb_array")
    num_actions = env.action_space.n
    
    print("num_actions: ", num_actions)

    # Create Q-network
    model = create_model(window, input_shape, num_actions).to(device)
    ### add load model
    ### XINHANG ###
    model.load_state_dict(torch.load('model2.pth'))
    print("Model loaded")
    # Create replay memory
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, window)
    preprocessor = Preprocessor(window = window)
    base_policy = GreedyEpsilonPolicy(0.05)
    policy = LinearDecayGreedyEpsilonPolicy(base_policy, 'epsilon', 1, 0.001, 1000000)
    num_burn_in = 10000
    train_freq = 10
    num_actions = env.action_space.n
    # Create DQN agent
    agent = DQNAgent(model, preprocessor, replay_memory, policy, GAMMA,\
                    TARGET_UPDATE_FREQ, num_burn_in,train_freq, BATCH_SIZE, num_actions, device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08, \
                                     weight_decay=WEIGHT_DECAY, momentum=0, centered=False)
    loss_func = mean_huber_loss
    agent.compile(optimizer, loss_func)


    NUM_EPISODES = 100000000
    # Train agent
    agent.fit(env, NUM_EPISODES , max_episode_length=10000)





  
    
