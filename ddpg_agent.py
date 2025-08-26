from models import Actor, Critic
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ReplayBuffer import ReplayBuffer
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2,
                 buffer_size, batch_size, gamma,
                 tau, lr_actor, lr_critic, weight_decay, learn_every, num_training, random_seed):
        """Initialize an DDPG agent."""
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.learn_every = learn_every
        self.seed = random.seed(random_seed)

        self.t_step = 0      # Initialize time step (for updating every UPDATE_EVERY steps)
        self.num_training = num_training


        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, hidden_size_1, hidden_size_2, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_size_1, hidden_size_2, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.actor_target.load_state_dict(self.actor_local.state_dict())

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, hidden_size_1, hidden_size_2, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, hidden_size_1, hidden_size_2, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, random_seed)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().squeeze(0)
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.learn_every

        if self.t_step == 0:
            for i in range(0, self.num_training):
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences

        # Set target 
        actions_next = self.actor_target(next_states)
        Q_target_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_target = rewards + (self.gamma * Q_target_next * (1 - dones))

        # Update the critic by minimizing the loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Update the actor using the sampled policy gradient
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update the target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        

    def soft_update(self, local_model, target_model, tau=1e-3):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def noise_reset(self):
        """Reset all noise processes."""
        self.noise.reset()

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size) # long-term mean
        self.theta = theta # rate of mean reversion (how fast the process returns to the mean)
        self.sigma = sigma # volatility (randomness)
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # This implicitly assumes dt = 1
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state