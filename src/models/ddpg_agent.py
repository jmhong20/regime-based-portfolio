import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .actor_critic import Actor, Critic

# Check if GPU is available
device = 3
if device == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
print("using: ", device)

# Set all random seeds
def set_seed(seed=42):
    """
    Sets the random seed for torch, numpy, and random for consistent results.

    Args:
        seed (int): The seed value to use for all random operations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Replay Buffer (converted from replay_buffer.py)
class ReplayBuffer:
    """
    A buffer to store experiences for the agent. 
    This allows the agent to learn from a batch of past experiences.
    
    Args:
        buffer_size (int): Maximum number of experiences the buffer can hold.
        batch_size (int): Number of experiences to sample when training.
    """
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, next_state, done, index_regime_prior, prev_prior):
        """
        Adds an experience to the buffer.
        
        Args:
            state: The state before the action was taken.
            action: The action taken by the agent.
            reward: The reward received for the action.
            next_state: The state after the action was taken.
            done: Boolean indicating if the episode ended.
        """
        experience = (state, action, reward, next_state, done, index_regime_prior, prev_prior)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self):
        """
        Randomly samples a batch of experiences from the buffer.

        Returns:
            tuple: Arrays of states, actions, rewards, next_states, and done flags.
        """
        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)
        states = np.array([_[0] for _ in batch])
        actions = np.array([_[1] for _ in batch])
        rewards = np.array([_[2] for _ in batch])
        next_states = np.array([_[3] for _ in batch])
        dones = np.array([_[4] for _ in batch])
        index_regime_priors = np.array([_[5] for _ in batch])
        prev_priors = np.array([_[6] for _ in batch])
        return states, actions, rewards, next_states, dones, index_regime_priors, prev_priors

    def size(self):
        """
        Returns the current size of the buffer.
        
        Returns:
            int: Number of experiences stored in the buffer.
        """
        return self.count


# Ornstein-Uhlenbeck Noise (converted from ornstein_uhlenbeck.py)
class OUNoise:
    """
    Implements Ornstein-Uhlenbeck process for generating noise, which is used to encourage exploration in continuous action spaces.

    Args:
        mu (float): Mean of the noise.
        sigma (float): Volatility of the noise.
        theta (float): Speed of mean reversion.
        dt (float): Time step increment.
        x0 (float, optional): Initial noise value.
    """
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Generates a sample of noise using the Ornstein-Uhlenbeck process.
        
        Returns:
            ndarray: The noise sample.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """
        Resets the noise value to the initial value or mean.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# DDPG Agent (converted from ddpg.py)
class DDPG:
    """
    Implements the DDPG agent, which is a model-free, off-policy actor-critic algorithm 
    that uses deep neural networks for continuous action spaces.
    
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self,
                 seed,
                 action_dim,
                 N, T, F):

        self.feature_type = 'ohlcv'

        self.actor = Actor(seed, N, T, F).to(device)
        self.actor_target = Actor(seed, N, T, F).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(action_dim=action_dim, seed=seed, N=N, T=T, F=F).to(device)
        self.critic_target = Critic(action_dim=action_dim, seed=seed, N=N, T=T, F=F).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(1e5, 64)
        self.noise = OUNoise(mu=np.zeros(action_dim))

        self.discount = 0.99
        self.tau = 0.001
        set_seed(seed)
        self.maxQ = None

    def normalize(self, x):
        return (x - 1) * 100

    def obs_normalizer(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]

        open_feature = self.normalize(observation[:, 1:, 0:1] / observation[:, 1:, 3:4])
        high_feature = self.normalize(observation[:, 1:, 1:2] / observation[:, 1:, 3:4])
        low_feature = self.normalize(observation[:, 1:, 2:3] / observation[:, 1:, 3:4])
        close_feature = self.normalize(observation[:, 1:, 3:4] / observation[:, :-1, 3:4])
        volume_feature = self.normalize(observation[:, 1:, 4:5] / observation[:, :-1, 4:5])

        if self.feature_type == 'cv':
            observation = np.concatenate((close_feature, volume_feature), axis=2)
        elif self.feature_type == 'open':
            observation = np.concatenate((open_feature), axis=2)
        elif self.feature_type == 'high':
            observation = np.concatenate((high_feature), axis=2)
        elif self.feature_type == 'low':
            observation = np.concatenate((low_feature), axis=2)
        elif self.feature_type == 'close':
            return close_feature
        elif self.feature_type == 'ohlc':
            observation = np.concatenate((open_feature, high_feature, low_feature, close_feature), axis=2)
        elif self.feature_type == 'ohlcv':
            observation = np.concatenate((open_feature, high_feature, low_feature, close_feature, volume_feature), axis=2)
        return observation

    def select_action(self, state, verbose=False,noise=None, prior=None):
        """
        Selects an action based on the current policy (actor network).

        Args:
            state (ndarray): The current state of the environment.

        Returns:
            ndarray: The selected action.
        """
        state = torch.FloatTensor(state).to(device)
        if prior is not None:
            prior = torch.FloatTensor(prior).to(device)
        final_portfolio, portfolio_weights, index_regime_prior, prior = self.actor(
                                                                        x=state,
                                                                        noise=noise,
                                                                        verbose=verbose,
                                                                        prior=prior
                                                                    )
        return final_portfolio.cpu().data.numpy().flatten(), portfolio_weights.cpu().data.numpy(), index_regime_prior.cpu().numpy(), prior.cpu().numpy()

    def train(self):
        if self.replay_buffer.size() < self.replay_buffer.batch_size:
            return

        # Sample a batch of transitions from replay buffer
        states, actions, rewards, next_states, dones, index_regime_priors, prev_priors = self.replay_buffer.sample()

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        index_regime_priors = torch.FloatTensor(index_regime_priors).to(device)
        prev_priors = torch.FloatTensor(prev_priors).to(device)

        # Compute the target Q value
        next_actions,_,_,_ = self.actor_target(x=next_states, prior=index_regime_priors)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (self.discount * target_Q * (1 - dones))

        # Get current Q estimate
        current_Q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        predicted_actions,_,_,_ = self.actor(x=states, prior=prev_priors)
        q_values_for_actor = self.critic(states, predicted_actions)
        max_q_value = q_values_for_actor.max().item()
        self.maxQ = max_q_value
        actor_loss = -q_values_for_actor.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_transition(self, transition):
        """
        Adds a transition to the replay buffer.
        
        Args:
            transition (tuple): A tuple containing (state, next_state, action, reward, done).
        """
        state, next_state, action, reward, done, index_regime_prior, prev_prior = transition
        self.replay_buffer.add(state, action, reward, next_state, done, index_regime_prior, prev_prior)

    def save_model(self, actor_path, critic_path):
        """
        Saves the actor and critic networks to the specified paths.

        Args:
            actor_path (str): Path to save the actor network.
            critic_path (str): Path to save the critic network.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(actor_path), exist_ok=True)
        os.makedirs(os.path.dirname(critic_path), exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        """
        Loads the actor and critic networks from the specified paths.

        Args:
            actor_path (str): Path to load the actor network from.
            critic_path (str): Path to load the critic network from.
        """
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
