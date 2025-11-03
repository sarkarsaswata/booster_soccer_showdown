import random
from collections import deque
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms.
    
    Stores transitions and allows sampling of random minibatches for training.
    
    Args:
        max_size: Maximum number of transitions to store
    """
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


def add_noise(action: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
    """Add Gaussian noise to actions for exploration.
    
    Args:
        action: Action array
        noise_scale: Standard deviation of noise
        
    Returns:
        Noisy action clipped to [-1, 1]
    """
    noise = np.random.normal(0, noise_scale, size=action.shape)
    return np.clip(action + noise, -1, 1)


def training_loop(
    env: gym.Env,
    model,
    action_function: Optional[Callable] = None,
    preprocess_class: Optional[Callable] = None,
    timesteps: int = 1000,
):
    """Main training loop for DDPG agent.
    
    Args:
        env: Gymnasium environment
        model: DDPG model to train
        action_function: Optional function to map policy outputs to environment actions
        preprocess_class: Optional preprocessor class for observations
        timesteps: Total number of environment steps to train for
    """
    replay_buffer = ReplayBuffer(max_size=100000)
    preprocessor = preprocess_class()
    batch_size = 64
    update_frequency = 4

    total_steps = 0
    episode_count = 0

    pbar = tqdm(total=timesteps, desc="Training Progress", unit="steps")

    while total_steps < timesteps:
        done = False
        s, info = env.reset()
        s = preprocessor.modify_state(s, info).squeeze()
        episode_reward = 0
        episode_steps = 0

        while not done and total_steps < timesteps:
            state = torch.from_numpy(np.expand_dims(s, axis=0))
            policy = model(state).detach().numpy()

            if action_function:
                action = action_function(policy)[0].squeeze()
                action = add_noise(action, noise_scale=0.1)
            else:
                action = model.select_action(s)[0].squeeze()

            new_s, r, terminated, truncated, info = env.step(action)
            new_s = preprocessor.modify_state(new_s, info).squeeze()

            done = terminated or truncated

            episode_reward += r
            episode_steps += 1

            replay_buffer.add(s, action, r, new_s, done)
            s = new_s

            total_steps += 1
            pbar.update(1)

            if len(replay_buffer) >= batch_size and total_steps % update_frequency == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    batch_size
                )
                critic_loss, actor_loss = model.train(
                    states,
                    actions,
                    rewards.reshape(-1, 1),
                    next_states,
                    dones.reshape(-1, 1),
                    1,
                )

                pbar.set_description(
                    f"Episode {episode_count} | Reward: {episode_reward:.2f} | Critic: {critic_loss:.4f} | Actor: {actor_loss:.4f}"
                )

        episode_count += 1

    pbar.close()
    env.close()
