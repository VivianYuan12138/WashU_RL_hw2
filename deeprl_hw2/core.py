"""Core classes."""
import random
from collections import deque
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import torch
class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    ### XINHANG ###
    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """
    ### XINHANG ###
    def __init__(self, new_size=(84, 84), history_length=4):
      self.new_size = new_size
      self.history_length = history_length
      self.state_buffer = deque(maxlen=history_length)
    
    # ### XINHANG ###
    # def process_state_for_network(self, state):
    #     """Preprocess the given state before giving it to the network.

    #     Should be called just before the action is selected.

    #     This is a different method from the process_state_for_memory
    #     because the replay memory may require a different storage
    #     format to reduce memory usage. For example, storing images as
    #     uint8 in memory is a lot more efficient thant float32, but the
    #     networks work better with floating point images.

    #     Parameters
    #     ----------
    #     state: np.ndarray
    #       Generally a numpy array. A single state from an environment.

    #     Returns
    #     -------
    #     processed_state: np.ndarray
    #       Generally a numpy array. The state after processing. Can be
    #       modified in anyway.

    #     """
    #     processed_state = resize(rgb2gray(state[0]), self.new_size).astype(np.float32)
    #     self.state_buffer.append(processed_state)
    #     # 确保状态缓冲区始终被填满
    #     while len(self.state_buffer) < self.history_length:
    #         self.state_buffer.append(processed_state)
    #     stacked_state = np.stack(self.state_buffer, axis=0)  # 堆叠成 (history_length, new_size[0], new_size[1])
    #     return stacked_state
    

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: torch.Tensor
          A tensor suitable for being an input to the network. The tensor shape is (N, C, H, W).

        """
        # 首先处理当前状态并将其添加到状态缓冲区
        processed_state = resize(rgb2gray(state), self.new_size).astype(np.float32)
        self.state_buffer.append(processed_state)

        # 如果状态缓冲区未满，用第一个帧填充
        while len(self.state_buffer) < self.history_length:
            self.state_buffer.append(processed_state)

        # 将状态缓冲区中的帧堆叠成一个 NumPy 数组
        stacked_state = np.stack(self.state_buffer, axis=0)  # 堆叠成 (C, H, W)

        # 转换为 PyTorch Tensor 并添加一个新的批次维度 (N, C, H, W)，其中 N=1
        stacked_state = torch.tensor(stacked_state).unsqueeze(0)

        return stacked_state

    ### XINHANG ###
    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return np.uint8(resize(rgb2gray(state), self.new_size) * 255)

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        # 批量处理，不更新状态历史记录
        processed_states = np.array([resize(rgb2gray(state), self.new_size).astype(np.float32) for state in samples])
        return processed_states

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return np.clip(reward, -1., 1.)

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        self.state_buffer.clear()

    ### XINHANG ###
    def add_state(self, state):
      processed_state = self.process_state_for_network(state)
      self.state_buffer = np.roll(self.state_buffer, -1, axis=2)
      self.state_buffer[:, :, -1] = processed_state


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        ### XINHANG ###
        self.max_size = max_size
        self.window_length = window_length
        self.memory = deque(maxlen=max_size)

    def append(self, state, action, reward, next_state, done):
        """Add a sample to the replay memory.

        Parameters
        ----------
        state: np.ndarray
          The state of the environment
        action: int, float, tuple
          The action taken in the state
        reward: float
          The reward received from taking the action
        """
        ### XINHANG ###
        self.memory.append((state, action, reward, next_state, done))
        

    def end_episode(self, final_state, is_terminal):
        """Set the final state of an episode.

        Parameters
        ----------
        final_state: np.ndarray
          The final state of the episode
        is_terminal: bool
          True if the episode ended in a terminal state. False
          otherwise.
        """
        ### XINHANG ###
        self.memory[-1] = (self.memory[-1][0], self.memory[-1][1], self.memory[-1][2], final_state, is_terminal)

    def sample(self, batch_size, indexes=None):
        """Return list of samples from the memory.

        Each class will implement a different method of choosing the
        samples. Optionally, specify the sample indexes manually.

        Parameters
        ----------
        batch_size: int
          The number of samples to return
        indexes: list(int)
          A list of indexes to return. If this is not None, then
          batch_size is ignored.

        Returns
        -------
        samples: list(tensorflow_rl.core.Sample)
          A list of samples from the memory. The length of the list
          should be batch_size.
        """
        ### XINHANG ###
        if indexes is not None:
            return [self.memory[i] for i in indexes]
        else:
            return random.sample(self.memory, batch_size)

    # def sample_batch(self, batch_size):
    #     batch = []
    #     while len(batch) < batch_size:
    #         # 随机选择一个起始索引，保证有足够的帧可以形成一个完整的序列
    #         index = random.randint(0, len(self.memory) - self.window_length)
    #         # 检查选取的序列是否跨越了回合的边界
    #         if any(self.memory[index + i][4] for i in range(self.window_length - 1)):
    #             continue  # 如果在序列中间发现结束帧，则跳过此序列

    #         # 构建状态和下一状态的序列
    #         states = [self.memory[index + i][0] for i in range(self.window_length)]
    #         next_states = [self.memory[index + i][3] for i in range(self.window_length)]
    #         actions = [self.memory[index + i][1] for i in range(self.window_length)]
    #         rewards = [self.memory[index + i][2] for i in range(self.window_length)]
    #         dones = [self.memory[index + i][4] for i in range(self.window_length)]

    #         # 将状态序列转换为适当的张量格式
    #         state_batch = torch.tensor(states, dtype=torch.float32)
    #         next_state_batch = torch.tensor(next_states, dtype=torch.float32)
    #         action_batch = torch.tensor(actions, dtype=torch.int64)
    #         reward_batch = torch.tensor(rewards, dtype=torch.float32)
    #         done_batch = torch.tensor(dones, dtype=torch.float32)
            

    #         batch.append((state_batch, action_batch, reward_batch, next_state_batch, done_batch))

    #     return batch
        
    def sample_batch(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            # Randomly choose a starting index
            start = random.randint(0, len(self.memory) - self.window_length)
            # Check if the sequence crosses an episode boundary
            if any(self.memory[i][4] for i in range(start, start + self.window_length - 1)):
                continue  # Skip if any state in the sequence (except the last) is terminal
           
            # Construct the state and next_state sequences
            states = [self.memory[i][0] for i in range(start, start + self.window_length)]
            next_states = [self.memory[i][3] for i in range(start, start + self.window_length)]
            # Get the other elements of the transition from the last sample in the sequence
            action = self.memory[start + self.window_length - 1][1]
            reward = self.memory[start + self.window_length - 1][2]
            done = self.memory[start + self.window_length - 1][4]   

            # Convert the state sequences to the appropriate tensor format
            state_batch = torch.tensor(np.array(states), dtype=torch.float32)
            next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32)
            action_batch = torch.tensor(action, dtype=torch.int64)
            reward_batch = torch.tensor(reward, dtype=torch.float32)
            done_batch = torch.tensor(done, dtype=torch.float32)

            batch.append((state_batch, action_batch, reward_batch, next_state_batch, done_batch))
        return batch


    def clear(self):
        """Reset the memory.

        Deletes all references to the samples.
        """
        ### XINHANG ###
        self.memory.clear()
    
    ### XINHANG ###
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, index):
        return self.memory[index]
    
    def __iter__(self):
        return iter(self.memory)