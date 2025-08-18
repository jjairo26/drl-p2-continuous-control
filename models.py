import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, action_size)

        self.reset_parameters()

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action = F.tanh(self.fc3(x)) # To limit output values
        return action

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_state_size, hidden_action_size):
        # This network has the states as an input in the input layer to determine state features
        # The actions come as an input starting from the first hidden layer
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_state_size)
        self.fc2 = nn.Linear(hidden_state_size + action_size, hidden_action_size)
        self.fc3 = nn.Linear(hidden_action_size, 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(torch.cat([x, action], dim=1)) 
        x = F.relu(x)
        value = self.fc3(x)
        return value
    