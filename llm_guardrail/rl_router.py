import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
from collections import deque
import random
import json
import os
from datetime import datetime
from pathlib import Path

class ModelPerformanceEnvironment:
    """Environment for the RL agent to learn model selection."""
    
    def __init__(self, models_config: Dict[str, Dict[str, Any]]):
        self.models = models_config
        self.state_size = len(self._get_state_features({}))  # Initialize with empty state
        self.action_size = len(models_config)
        self.model_names = list(models_config.keys())
        
    def _get_state_features(self, prompt_data: Dict[str, Any]) -> np.ndarray:
        """Extract relevant features from prompt and context."""
        features = [
            len(prompt_data.get('prompt', '')),  # Prompt length
            float(prompt_data.get('estimated_complexity', 0.5)),  # Task complexity
            float(prompt_data.get('token_estimate', 0)),  # Estimated tokens
            float(prompt_data.get('time_sensitivity', 0.5)),  # Time sensitivity
            float(prompt_data.get('cost_sensitivity', 0.5)),  # Cost sensitivity
        ]
        return np.array(features, dtype=np.float32)
    
    def reset(self, prompt_data: Dict[str, Any]) -> np.ndarray:
        """Reset environment with new prompt data."""
        self.current_state = self._get_state_features(prompt_data)
        return self.current_state
    
    def step(self, action: int, result: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return new state, reward, done flag, and info."""
        model_name = self.model_names[action]
        model_spec = self.models[model_name]
        
        # Calculate reward based on multiple factors
        success_reward = 1.0 if result['status'] == 'success' else -1.0
        retry_penalty = -0.2 * result.get('retry_count', 0)
        
        # Latency reward (negative correlation)
        latency = result.get('latency', model_spec['latency'])
        latency_reward = -0.1 * (latency / max(m['latency'] for m in self.models.values()))
        
        # Cost reward (negative correlation)
        cost = result.get('cost', model_spec['cost_per_token'])
        cost_reward = -0.1 * (cost / max(m['cost_per_token'] for m in self.models.values()))
        
        total_reward = success_reward + retry_penalty + latency_reward + cost_reward
        
        return self.current_state, total_reward, True, {
            'model_used': model_name,
            'success': result['status'] == 'success',
            'retries': result.get('retry_count', 0)
        }

class DQNAgent:
    """Deep Q-Network agent for learning optimal model selection."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action based on epsilon-greedy policy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def replay(self, batch_size: int):
        """Train on random batch from memory."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path))

class RLRouter:
    """RL-based router that learns to select optimal models."""
    
    def __init__(self, models_config: Dict[str, Dict[str, Any]], save_dir: str = "model_data"):
        self.models_config = models_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize environment and agent
        self.env = ModelPerformanceEnvironment(models_config)
        self.agent = DQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size
        )
        
        # Try to load existing model
        model_path = self.save_dir / "rl_router_model.pth"
        if model_path.exists():
            self.agent.load(str(model_path))
            self.agent.epsilon = self.agent.epsilon_min  # Set to minimum for production
    
    def select_model(self, prompt_data: Dict[str, Any]) -> str:
        """Select best model based on current policy."""
        state = self.env.reset(prompt_data)
        action = self.agent.act(state)
        return self.env.model_names[action]
    
    def update(self, prompt_data: Dict[str, Any], result: Dict[str, Any]):
        """Update agent's knowledge based on execution result."""
        state = self.env.reset(prompt_data)
        action = self.env.model_names.index(result['model_used'])
        
        # Get reward from environment
        _, reward, done, _ = self.env.step(action, result)
        
        # Store experience
        next_state = state  # In our case, episode ends after one step
        self.agent.remember(state, action, reward, next_state, done)
        
        # Train on a batch
        self.agent.replay(batch_size=32)
        
        # Save periodically (can be optimized based on your needs)
        self.agent.save(str(self.save_dir / "rl_router_model.pth"))
        
        # Log performance data
        self._log_performance(prompt_data, result, reward)
    
    def _log_performance(self, prompt_data: Dict[str, Any], result: Dict[str, Any], reward: float):
        """Log performance data for analysis."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'prompt_length': len(prompt_data.get('prompt', '')),
            'estimated_complexity': prompt_data.get('estimated_complexity', 0.5),
            'model_used': result['model_used'],
            'status': result['status'],
            'retry_count': result.get('retry_count', 0),
            'latency': result.get('latency'),
            'reward': reward
        }
        
        log_path = self.save_dir / "performance_log.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n') 