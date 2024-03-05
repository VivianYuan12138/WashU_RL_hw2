import torch
import torch.nn as nn

class DQN(nn.Module):
    """DQN Architecture as described in the DQN paper."""
    def __init__(self, input_shape, num_actions):
        """
        Initialize the DQN network.

        Parameters:
        - input_shape: The shape of the input observations. For Atari, it's (C, H, W) where C is the channel (stacked frames), H is the height, and W is the width of the image.
        - num_actions: The number of possible actions in the environment.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        - x: Input tensor of shape (N, C, H, W) where N is the batch size.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def _feature_size(self, input_shape):
        """
        Determine the size of the features after the convolutional layers.
        
        Parameters:
        - input_shape: The shape of the input observations.
        """
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

# Example usage:
input_shape = (4, 84, 84)  # 4 stacked frames of 84x84 pixels
num_actions = 4  # Example number of actions in an environment
model = DQN(input_shape, num_actions)
print(model)
