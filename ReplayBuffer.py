from collections import deque
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64, random_seed=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.random_seed = random_seed
        random.seed(random_seed)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)

        states = torch.from_numpy(np.vstack([b[0] for b in batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([b[1] for b in batch])).long().to(device)
        rewards = torch.from_numpy(np.vstack([b[2] for b in batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([b[4] for b in batch]).astype(np.uint8)).float().to(device) # True/False to 1/0, then to float
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)