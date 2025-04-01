import torch
import torch.nn as nn
import torch.optim as optim
from models.q_network import QNetwork
from .replay_buffer import ReplayBuffer
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_buffer = ReplayBuffer(10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已載入自 {filepath}")

    def act(self, state):
        """根據當前狀態選擇動作 (ε-greedy)"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def train(self):
        """從 Replay Buffer 取樣進行 Q-learning 訓練"""
        if self.replay_buffer.size() < 32:
            return  # 不足批次大小，跳過訓練
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(32)
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        q_values_next = self.target_model(next_states)

        # 計算目標 Q 值
        target_q_values = q_values.clone()
        for i in range(32):
            target_q_values[i, actions[i]] = rewards[i] + (1 - dones[i]) * 0.99 * torch.max(q_values_next[i])

        # 計算損失並反向傳播
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 更新目標網路
        self.target_model.load_state_dict(self.model.state_dict())

if __name__ == "__main__":
    agent = DQNAgent(state_size=16, action_size=4)
    state = np.zeros(16)
    action = agent.act(state)
    print(f"Selected action: {action}")