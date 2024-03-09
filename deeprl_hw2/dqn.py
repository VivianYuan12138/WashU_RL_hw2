import numpy as np
import torch.optim as optim
import copy
from deeprl_hw2.utils import get_soft_target_model_updates, get_hard_target_model_updates
import torch
from collections import deque
from torch import Tensor
from gymnasium.wrappers import RecordVideo

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network:
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    policy: deeprl_hw2.policy.Policy
      Your action selection policy.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
      If target_update_freq > 0, it represents the number of steps between hard updates.
      If target_update_freq < 0, its absolute value can be used as the τ value in soft updates.
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    device: str or torch.device
      The device to move the model and tensors to, e.g., 'cuda' or 'cpu'.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 num_actions,
                 device
                ):
        self.q_network = q_network.to(device)
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.steps = 0
        self.num_actions = num_actions
        self.target_network = self._init_target_network(q_network).to(device)
        self.device = device
        self.window = self.preprocessor.window

        with open('log_a.txt', 'a') as f:
            f.write('')



    def _init_target_network(self, q_network):
        target_network = copy.deepcopy(q_network)
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        return target_network


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.optimizer = optimizer
        self.loss_func = loss_func

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """

  
        return self.target_network(state)

    def select_action(self, state, train=True):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        
        return self.policy.select_action(self.calc_q_values(state).cpu().detach().numpy(), is_training=train)
    
        # return self.policy.select_action(self.calc_q_values(state).cpu().detach().numpy())


        # q_values = self.calc_q_values(state).detach().numpy()
        # return np.argmax(q_values)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample_batch(self.batch_size)
        # transitions = self.memory.sample_batch_reward_isnot0(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert tuples to tensors
        states = torch.stack(states).to(self.device) #[32,4,84,84]
        next_states = torch.stack(next_states).to(self.device) #[32,4,84,84]
        actions = torch.stack(actions).to(self.device) #[32]
        rewards = torch.stack(rewards).to(self.device) #[32]
        dones = torch.stack(dones).to(self.device) #[32]

        # Compute current Q values for each action taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) #[32]

        # Compute max Q values for next states from the target network
        next_q_values = self.target_network(next_states).max(1)[0] #[32]

        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones) #[32]

        # Compute loss
        loss = self.loss_func(current_q_values, expected_q_values.detach())
        
        print("loss: ", loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        if self.target_update_freq > 0 and self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        elif self.target_update_freq < 0:
            tau = abs(self.target_update_freq)
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.target_network.eval()

        self.steps += 1

        

    def fit(self, env, num_iterations, max_episode_length=None):
        for i in range(num_iterations):
            print("Episode: ", i+1)
            state = env.reset()
            self.preprocessor.reset()  # Reset the preprocessor for the new episode
            episode_reward = 0
            episode_length = 0  # Initialize episode length
            initial_lives = 3
            step_record = []
            while True:
                # Process the current state for the network, updating the state history
                processed_state = self.preprocessor.process_state_for_network(state[0]).to(self.device)
                
                action = self.select_action(processed_state)  # Select an action based on the processed state
                if len(step_record) <= 50:
                    step_record.append(action)
                if len(step_record) == 50:
                    print("step_record: ", step_record)
                    with open('log_a.txt', 'a') as f:
                        f.write(f"step_record: {step_record}\n")
                next_state, reward, done, truncated, info = env.step(action)  # Take the action in the environment
                ### info format: {'lives': 3, 'episode_frame_number': 5, 'frame_number': 5}
                # print("info: ", info)
                #### MODIFY THE REWARD HERE###
                
                ###1. negative reward for staying still
                
                #print("action: ", action)
                
                #clip reward to -1, 1
                # reward = np.clip(reward, -1, 1)
                
                ###
                #current_lives = info['lives']
                #if current_lives < initial_lives:
                #    reward = - 100
                #    initial_lives = current_lives

                episode_reward += reward
                episode_length += 1  # Increment episode length

                
                # Process the next state for memory, without updating the state history
                processed_state = self.preprocessor.process_state_for_memory(state[0])
                processed_next_state = self.preprocessor.process_state_for_memory(next_state)
                
                # Add the transition to the replay memory
                self.memory.append(processed_state, action, reward, processed_next_state, done)

                self.steps += 1  # Increment the number of steps taken
                # Update policy if conditions are met
                if self.steps > self.num_burn_in and self.steps % self.train_freq == 0:
                    self.update_policy()

                state = next_state  # Move to the next state
                # Check if the episode should end
                if done or (max_episode_length and episode_length >= max_episode_length):
                    break

            # print(f"Episode {i+1}: Reward: {episode_reward}, Length: {episode_length}, Steps: {self.steps}, Memory Length: {len(self.memory)}")
            with open('log_a.txt', 'a') as f:
                f.write(f"Episode {i+1}: Reward: {episode_reward}, Length: {episode_length}, Steps: {self.steps}, Memory Length: {len(self.memory)}\n")

            # Periodically evaluate the agent's performance
            if i % 20 == 0:
                self.evaluate(env, 5, max_episode_length)

        env.close()

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment."""
        total_rewards = []  # To store total rewards for each episode

        for episode in range(num_episodes):
            state = env.reset()
            self.preprocessor.reset()  # Reset the preprocessor for the new episode
            episode_reward = 0
            episode_length = 0  # Initialize episode length
            done = False

            while not done:
                # Process the current state for the network, updating the state history
                processed_state = self.preprocessor.process_state_for_network(state[0]).to(self.device)

                action = self.select_action(processed_state, train=False)  # Select an action based on the processed state
                next_state, reward, done, _ , _ = env.step(action)  # Take the action in the environment

                episode_reward += reward
                episode_length += 1  # Increment episode length

                state = next_state  # Move to the next state

                # Check if the episode should end
                if max_episode_length and episode_length >= max_episode_length:
                    done = True

            total_rewards.append(episode_reward)  # Store total reward for this episode

        average_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {average_reward}")
        with open('log_a.txt', 'a') as f:
            f.write(f"Average Reward over {num_episodes} episodes: {average_reward}\n")
        # save the model
        torch.save(self.q_network.state_dict(), 'model2.pth')
        return average_reward
    
    def record_video(self, env, model_path, output_directory, num_episodes=1, max_episode_length=None):
        """
        Record the agent playing with the model loaded from the given path.

        Parameters:
        - env: The Gym environment to play in.
        - model_path: Path to the model weights.
        - output_directory: Directory where the recorded videos will be saved.
        - num_episodes: Number of episodes to record.
        - max_episode_length: Optional limit on the number of steps per episode.
        """
        # Load model weights
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.q_network.to(self.device)
        self.q_network.eval()  # Set the model to evaluation mode

        self.target_network = self._init_target_network(self.q_network).to(self.device)

        # open record_file.txt
        with open('record_file.txt', 'a') as f:
            f.write('')
        # Wrap the environment to record video
        env = RecordVideo(env, video_folder=output_directory, episode_trigger=lambda episode: True)
        total_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            self.preprocessor.reset()  # Reset the preprocessor for the new episode
            episode_reward = 0
            episode_length = 0  # Initialize episode length
            done = False

            while not done:
                # Process the current state for the network, updating the state history
                processed_state = self.preprocessor.process_state_for_network(state[0]).to(self.device)

                action = self.select_action(processed_state, train=False)  # Select an action based on the processed state
                next_state, reward, done, _ , _ = env.step(action)  # Take the action in the environment

                episode_reward += reward
                episode_length += 1  # Increment episode length

                state = next_state  # Move to the next state

                # Check if the episode should end
                if max_episode_length and episode_length >= max_episode_length:
                    done = True

            total_rewards.append(episode_reward)  # Store total reward for this episode
            print(f"Episode {episode}: Reward: {episode_reward}, Length: {episode_length}")
            path = "record.txt"
            with open(path, 'a') as f:
                f.write(f"Episode {episode}: Reward: {episode_reward}, Length: {episode_length}\n")
        #also get the max index of the total_rewards
        max_index = np.argmax(total_rewards)
        average_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)  
        print(f"Average Reward over {num_episodes} episodes: {average_reward}, std: {std_reward}, max_index: {max_index}")
        with open('record_file.txt', 'a') as f:
            f.write(f"Average Reward over {num_episodes} episodes: {average_reward}, std: {std_reward}\n")
    