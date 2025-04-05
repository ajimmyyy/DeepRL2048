import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml
import numpy as np
from models.q_network import QNetwork
from models.dueling_q_network import DuelingQNetwork
from .replay_buffer import PrioritizedReplayBuffer
from utils.config import Config

#設定
config = Config()

class DQNAgent:
    def __init__(self, state_size, action_size, device_type="cpu"):
        # 選擇裝置
        self.device = self.get_device(device_type)
        torch.backends.cudnn.benchmark = True

        self.model = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_model = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.action_size = config.ACTION_SIZE
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.target_update_frequency = config.TARGET_UPDATE_FREQUENCY

        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = PrioritizedReplayBuffer(config.REPLAY_BUFFER_SIZE, config.ALPHA, config.BETA, config.BETA_INCREMENT, config.PER_EPSILON, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.training_step = 0

    def get_device(self, device_type):
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "directml":
            return torch_directml.device()
        else:
            return torch.device("cpu")

    def save_model(self, filepath, model_name):
        """儲存模型的狀態"""
        config_summary = config.get_summary()

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'additional_info': config_summary,
        }, os.path.join(filepath, model_name))

        print(f"模型 {model_name} 已儲存到 {os.path.join(filepath, model_name)}")

    def load_model(self, filepath, model_name):
        """載入模型的狀態"""
        filepath = os.path.join(filepath, model_name)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已載入自 {filepath}")

    def act(self, states):
        """根據當前狀態選擇動作 (ε-greedy)"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size, size=len(states))

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(states_tensor)

        return torch.argmax(q_values, dim=1).cpu().numpy()

    def train(self):
        """從 Replay Buffer 取樣進行 Q-learning 訓練"""
        if self.replay_buffer.size() < self.batch_size:
            return
        
        idxs, states, actions, rewards, next_states, dones, weights = self.replay_buffer.sample(self.batch_size)

        # Q值計算
        q_values = self.model(states)
        q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_values_next = self.target_model(next_states)
            actions_next = self.model(next_states).argmax(dim=1)
            q_values_next_selected = q_values_next.gather(1, actions_next.unsqueeze(1)).squeeze(1)

        q_values_target = rewards + (1 - dones) * self.gamma * q_values_next_selected

        # 計算 TD 誤差
        td_errors = q_values_selected - q_values_target

        # Huber loss
        loss = (F.smooth_l1_loss(q_values_selected, q_values_target, reduction='none') * weights).mean()

        # 計算損失並反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 PER 優先級
        self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())

        # 指數衰減epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 每 N 次更新目標網路
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

