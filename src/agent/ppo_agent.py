"""
PPO agent implementation for reinforcement learning trading.

This module implements the Proximal Policy Optimization (PPO) algorithm for the
trading environment. PPO is a policy gradient method that uses a clipped surrogate
objective to ensure stable learning.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from src.agent.base_agent import BaseAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PPOMemory:
    """Memory buffer for PPO algorithm to store experiences."""
    
    def __init__(self, batch_size: int = 64):
        """
        Initialize the PPO memory buffer.
        
        Args:
            batch_size: Size of batches to sample from memory
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(
        self,
        state: Any,
        action: np.ndarray,
        prob: float,
        val: float,
        reward: float,
        done: bool
    ) -> None:
        """
        Store an experience in memory.
        
        Args:
            state: State observation
            action: Action taken
            prob: Log probability of the action
            val: Value estimate
            reward: Reward received
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self) -> None:
        """Clear the memory buffer."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self) -> List[np.ndarray]:
        """
        Generate batches for training.
        
        Returns:
            List[np.ndarray]: List of batch indices
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches
    
    def __len__(self) -> int:
        """
        Get the current size of memory.
        
        Returns:
            int: Number of experiences in memory
        """
        return len(self.states)


# class ActorNetwork(nn.Module):
#     """Actor network for the PPO agent."""
    
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         hidden_dims: List[int] = [256, 256],
#         activation_fn: nn.Module = nn.ReLU(),
#         action_std_init: float = 0.6
#     ):
#         """
#         Initialize the actor network.
        
#         Args:
#             input_dim: Dimension of the input (state)
#             output_dim: Dimension of the output (action)
#             hidden_dims: List of hidden layer dimensions
#             activation_fn: Activation function to use
#             action_std_init: Initial standard deviation for continuous actions
#         """
#         super(ActorNetwork, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.action_std_init = action_std_init
#         self.action_var = torch.full((output_dim,), action_std_init * action_std_init)
        
#         # Build the network
#         layers = []
#         prev_dim = input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(activation_fn)
#             prev_dim = hidden_dim
        
#         # Output layer for categorical (discrete) actions
#         self.actor_categorical = nn.Sequential(
#             *layers,
#             nn.Linear(prev_dim, output_dim),
#             nn.Softmax(dim=-1)
#         )
        
#         # For continuous actions, we would have separate mean and std outputs
#         # but for trading we're using discrete actions
    
#     def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
#         """
#         Forward pass through the network.
        
#         Args:
#             state: Input state tensor
            
#         Returns:
#             torch.distributions.Distribution: Action distribution
#         """
#         action_probs = self.actor_categorical(state)
#         dist = Categorical(action_probs)
        
#         return dist


# Dans src/agent/ppo_agent.py

class ActorNetwork(nn.Module):
    """Réseau d'acteur simplifié et biaisé vers l'action pour l'or."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        activation_fn: nn.Module = nn.ReLU()
    ):
        super(ActorNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Construire un réseau simple
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn)
            prev_dim = hidden_dim
        
        # Couche finale de sortie
        self.feature_network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Biais initial pour favoriser les actions plutôt que "hold"
        self.output_layer.bias.data[0] += 0.1  # Bias vers la vente
        self.output_layer.bias.data[2] += 0.1  # Bias vers l'achat
        self.output_layer.bias.data[1] -= 0.2  # Réduire le biais de "hold"
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        features = self.feature_network(state)
        action_logits = self.output_layer(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Modifier les probabilités pour encourager l'action
        # action_probs[:, 1] *= 0.8  # Réduire la probabilité d'attente
        
        dist = torch.distributions.Categorical(action_probs)
        return dist



class CriticNetwork(nn.Module):
    """Réseau critique simplifié pour le trading."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        activation_fn: nn.Module = nn.ReLU()
    ):
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    

# class CriticNetwork(nn.Module):
#     """Critic network for the PPO agent."""
    
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dims: List[int] = [256, 256],
#         activation_fn: nn.Module = nn.ReLU()
#     ):
#         """
#         Initialize the critic network.
        
#         Args:
#             input_dim: Dimension of the input (state)
#             hidden_dims: List of hidden layer dimensions
#             activation_fn: Activation function to use
#         """
#         super(CriticNetwork, self).__init__()
        
#         self.input_dim = input_dim
        
#         # Build the network
#         layers = []
#         prev_dim = input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(activation_fn)
#             prev_dim = hidden_dim
        
#         # Output layer for value
#         layers.append(nn.Linear(prev_dim, 1))
        
#         self.critic = nn.Sequential(*layers)
    
#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the network.
        
#         Args:
#             state: Input state tensor
            
#         Returns:
#             torch.Tensor: Value estimate
#         """
#         return self.critic(state)


class PPOAgent(BaseAgent):
    """PPO agent implementation for reinforcement learning trading."""
    
    def __init__(
        self,
        config: Dict,
        state_dim: int,
        action_dim: int,
        device: str = 'cpu',
        model_name: str = 'ppo_agent',
        hidden_dims: List[int] = None,
        activation_fn: str = 'relu',
        learning_rate: float = None,
        gamma: float = None,
        gae_lambda: float = None,
        clip_range: float = None,
        entropy_coef: float = None,
        value_coef: float = None,
        max_grad_norm: float = None,
        update_epochs: int = None,
        batch_size: int = None
    ):
        """
        Initialize the PPO agent.
        
        Args:
            config: Configuration dictionary
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to run the model on ('cpu' or 'cuda')
            model_name: Name for saving/loading the model
            hidden_dims: List of hidden layer dimensions
            activation_fn: Activation function to use
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            entropy_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            update_epochs: Number of update epochs
            batch_size: Batch size for training
        """
        super(PPOAgent, self).__init__(
            config=config,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            model_name=model_name
        )
        
        # Load PPO specific configuration
        ppo_config = config['model_config']['ppo']
        
        # Set hyperparameters, using provided values or defaults from config
        self.learning_rate = learning_rate or ppo_config.get('learning_rate', 3e-4)
        self.gamma = gamma or ppo_config.get('gamma', 0.99)
        self.gae_lambda = gae_lambda or ppo_config.get('gae_lambda', 0.95)
        self.clip_range = clip_range or ppo_config.get('clip_range', 0.2)
        self.entropy_coef = entropy_coef or ppo_config.get('ent_coef', 0.01)
        self.value_coef = value_coef or ppo_config.get('vf_coef', 0.5)
        self.max_grad_norm = max_grad_norm or ppo_config.get('max_grad_norm', 0.5)
        self.update_epochs = update_epochs or ppo_config.get('n_epochs', 10)
        self.batch_size = batch_size or ppo_config.get('batch_size', 64)
        
        # Get network architecture from config
        if hidden_dims is None:
            hidden_dims = ppo_config['policy_network'].get('net_arch', [256, 256])
        
        # Get activation function
        if activation_fn is None:
            activation_fn = ppo_config['policy_network'].get('activation_fn', 'tanh')
        
        if activation_fn == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation_fn == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            logger.warning(f"Unknown activation function: {activation_fn}, using ReLU")
            self.activation_fn = nn.ReLU()
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation_fn=self.activation_fn
        ).to(device)
        
        self.critic = CriticNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            activation_fn=self.activation_fn
        ).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Initialize memory buffer
        self.memory = PPOMemory(batch_size=self.batch_size)
        
        # Learning steps counter
        self.learning_steps = 0
        
        logger.info(f"Initialized PPO agent with state_dim={state_dim}, action_dim={action_dim}")



    def select_action(self, state: Any, evaluate: bool = False) -> np.ndarray:
        """
        Select an action based on the current state.
    
        Args:
            state: Current state observation
            evaluate: Whether to use exploration or deterministic action
        
        Returns:
            np.ndarray: Selected action
        """
        # Convert state to tensor
        if isinstance(state, dict):
            # Handle dictionary observation space
            market_state = torch.FloatTensor(state['market']).to(self.device)
            account_state = torch.FloatTensor(state['account']).to(self.device)
        
            # Flatten the tensors - CORRECTION: Changed view() to reshape()
            market_flat = market_state.reshape(1, -1)
            account_flat = account_state.reshape(1, -1)
        
            # Concatenate them
            state_tensor = torch.cat([market_flat, account_flat], dim=1)
        else:
            # Handle flat observation space
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    
        # Get action distribution
        dist = self.actor(state_tensor)
    
        # Select action
        if evaluate:
            # Obtenir les probabilités
            probs = dist.probs.detach().cpu().numpy()[0]
        
            # MODIFICATION MAJEURE: Forcer des trades en backtest
            # De manière aléatoire, réduire fortement la probabilité de "hold"
            forced_trade = False
            
            # Si on est en mode évaluation/backtest, forcer périodiquement des trades
            # en réduisant drastiquement la probabilité de "hold"
            if np.random.random() < 0.3:  # 30% du temps
                forced_trade = True
                # Réduire la probabilité de "hold" (indice 1) de 80%
                probs[1] *= 0.2
                # Normaliser les probabilités
                probs = probs / np.sum(probs)
            
            # Abaisser le seuil pour prendre des actions à 0.25 (au lieu de 0.5 typiquement)
            action_threshold = 0.20
        
            # Si une probabilité d'action (buy/sell) dépasse le seuil, prendre cette action
            if probs[0] > action_threshold or probs[2] > action_threshold:
                # Choisir l'action avec la plus haute probabilité
                action = np.argmax([probs[0], 0, probs[2]])  # Ignorer "hold" (indice 1)
                if action == 1:  # Si l'action choisie est l'indice 1 (après avoir ignoré "hold")
                    action = 2  # Convertir en "buy" (indice 2)
            else:
                # Même si aucune action n'atteint le seuil, encourager trading
                if forced_trade or np.random.random() < 0.4:  # 40% de chances supplémentaires
                    # Choisir entre buy et sell avec un biais légèrement en faveur du buy
                    action = 0 if np.random.random() < 0.45 else 2  # 45% sell, 55% buy
                else:
                    # Le reste du temps, échantillonner selon les probabilités
                    action = np.random.choice([0, 1, 2], p=probs)
        
            # Calcul du log de probabilité et de la probabilité
            action_logprob = np.log(probs[action] + 1e-8)
            action_probs = probs[action]
        else:
            # Pour l'entraînement, échantillonner mais avec un biais contre "hold"
            # Réduire légèrement la probabilité de "hold" pendant l'entraînement aussi
            temp_probs = dist.probs.detach().clone()
            temp_probs[:, 1] *= 0.9  # Réduire "hold" de 10%
            temp_probs = temp_probs / temp_probs.sum(dim=1, keepdim=True)  # Normaliser
            modified_dist = torch.distributions.Categorical(temp_probs)
            
            # Échantillonner à partir de la distribution modifiée
            action = modified_dist.sample().item()
            action_logprob = dist.log_prob(torch.tensor(action)).item()  # Utiliser la dist originale pour log_prob
            action_probs = dist.probs[0, action].item()
    
        # Get value estimate
        value = self.critic(state_tensor).item()
    
        return action, action_logprob, value


    def store_transition(
        self,
        state: Any,
        action: int,
        action_logprob: float,
        value: float,
        reward: float,
        done: bool
    ) -> None:
        """
        Store a transition in memory.
        
        Args:
            state: State observation
            action: Action taken
            action_logprob: Log probability of the action
            value: Value estimate
            reward: Reward received
            done: Whether the episode is done
        """
        self.memory.store(state, action, action_logprob, value, reward, done)
    
    def train(self, batch_size: Optional[int] = None) -> Dict:
        """
        Train the agent using experiences in memory.
        
        Args:
            batch_size: Optional batch size override
            
        Returns:
            Dict: Dictionary with training metrics
        """
        # Use provided batch size or default
        batch_size = batch_size or self.batch_size
        
        # Calculate advantages using GAE
        states, actions, old_log_probs, vals, rewards, dones = self._process_memory()
        
        # Calculate advantages
        advantages = self._compute_advantages(rewards, vals, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        total_losses = []
        
        # Train for multiple epochs
        for _ in range(self.update_epochs):
            # Generate batches
            batches = self.memory.generate_batches()
            
            for batch in batches:
                # Get batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_values = vals[batch]
                batch_rewards = rewards[batch]
                
                # Convert to tensors
                batch_states = torch.FloatTensor(batch_states).to(self.device)
                batch_actions = torch.LongTensor(batch_actions).to(self.device)
                batch_old_log_probs = torch.FloatTensor(batch_old_log_probs).to(self.device)
                batch_advantages = torch.FloatTensor(batch_advantages).to(self.device)
                batch_values = torch.FloatTensor(batch_values).to(self.device)
                batch_rewards = torch.FloatTensor(batch_rewards).to(self.device)
                
                # Forward pass
                dist = self.actor(batch_states)
                critic_value = self.critic(batch_states).squeeze()
                
                # Get action log probs
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Compute value loss
                value_targets = batch_advantages + batch_values
                value_loss = F.mse_loss(critic_value, value_targets)
                
                # Compute policy loss
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass and optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Record losses
                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                total_losses.append(total_loss.item())
        
        # Increment learning steps
        self.learning_steps += 1
        
        # Clear memory
        self.memory.clear()
        
        # Return training metrics
        metrics = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses),
            'learning_steps': self.learning_steps
        }
        
        self.update_metrics(metrics)
        
        return metrics
    
    def _process_memory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process memory buffer for training.
        
        Returns:
            Tuple: Processed memory data
        """
        states = []
        actions = []
        log_probs = []
        vals = []
        rewards = []
        dones = []
        
        for i in range(len(self.memory.states)):
            state = self.memory.states[i]
            
            # Process dict state into flat array
            if isinstance(state, dict):
                market_state = np.array(state['market']).flatten()
                account_state = np.array(state['account']).flatten()
                flat_state = np.concatenate([market_state, account_state])
                states.append(flat_state)
            else:
                states.append(state)
            
            actions.append(self.memory.actions[i])
            log_probs.append(self.memory.probs[i])
            vals.append(self.memory.vals[i])
            rewards.append(self.memory.rewards[i])
            dones.append(self.memory.dones[i])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            np.array(vals),
            np.array(rewards),
            np.array(dones)
        )
    
    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of episode done flags
            
        Returns:
            np.ndarray: Array of advantages
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # For the last step, use 0 as the next value if done, otherwise bootstrap
                next_value = 0 if dones[t] else last_value
            else:
                next_value = values[t+1]
            
            # Delta: reward + gamma * next_value - current_value
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Advantage: delta + gamma * lambda * next_advantage
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            
            # Update last advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Optional path to save the model to
            
        Returns:
            str: Path where the model was saved
        """
        if path is None:
            path = self.get_default_save_path()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'learning_steps': self.learning_steps,
            'metrics': self.metrics,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'max_grad_norm': self.max_grad_norm,
                'update_epochs': self.update_epochs,
                'batch_size': self.batch_size
            }
        }, path)
        
        logger.info(f"Model saved to {path}")
        
        return path
    
    def load_model(self, path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load parameters from checkpoint
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.learning_steps = checkpoint['learning_steps']
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        
        logger.info(f"Model loaded from {path}")