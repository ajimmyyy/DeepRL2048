import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import numpy as np
from models.q_network import QNetwork
from .replay_buffer import ReplayBuffer
from utils.config import Config

#設定
config = Config()

class DQNAgent:
    def __init__(self, state_size, action_size, device_type="cpu"):
        # 選擇裝置
        self.device = self.get_device(device_type)
        torch.backends.cudnn.benchmark = True

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE, device=self.device)
        self.epsilon = config.EPSILON_START
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.training_step = 0

    def get_device(self, device_type):
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "directml":
            return torch_directml.device()
        else:
            return torch.device("cpu")

    def save_model(self, filepath):
        """儲存模型的狀態"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"模型已儲存到 {filepath}")

    def load_model(self, filepath):
        """載入模型的狀態"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已載入自 {filepath}")

    def act(self, states):
        """根據當前狀態選擇動作 (ε-greedy)"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(config.ACTION_SIZE, size=len(states))

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(states_tensor)

        return torch.argmax(q_values, dim=1).cpu().numpy()

    def train(self):
        """從 Replay Buffer 取樣進行 Q-learning 訓練"""
        if self.replay_buffer.size() < config.BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(config.BATCH_SIZE)

        # Q值計算
        q_values = self.model(states)
        q_values_selected = q_values[torch.arange(config.BATCH_SIZE, device=self.device), actions]

        with torch.no_grad():
            q_values_next = self.target_model(next_states)
            actions_next = self.model(next_states).argmax(dim=1)
            q_values_next_selected = q_values_next.gather(1, actions_next.unsqueeze(1)).squeeze(1)

        q_values_target = rewards + (1 - dones) * config.GAMMA * q_values_next_selected

        # 計算損失並反向傳播
        loss = self.loss_fn(q_values_selected, q_values_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 指數衰減epsilon
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

        # 每 N 次更新目標網路
        self.training_step += 1
        if self.training_step % config.TARGET_UPDATE_FREQUENCY == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.training_step % config.EPSILON_RESET_STEP == 0:
            self.epsilon = config.EPSILON_RESET
            print(f"重設 epsilon 至 {self.epsilon}")

