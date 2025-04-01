import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, device=None):
        """初始化 Replay Buffer
        :param buffer_size: 緩衝區的最大大小
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, experience):
        """將經驗加入緩衝區
        :param experience: tuple (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """從緩衝區中隨機選擇一個 batch
        :param batch_size: 抽樣的大小
        :return: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def size(self):
        """返回當前緩衝區的大小"""
        return len(self.buffer)

# 測試
if __name__ == "__main__":
    buffer = ReplayBuffer(100)

    for _ in range(10):
        experience = (np.zeros(16), random.randint(0, 3), random.random(), np.zeros(16), random.choice([True, False]))
        buffer.add(experience)

    states, actions, rewards, next_states, dones = buffer.sample(5)
    print("Sampled Batch:")
    print(states, actions, rewards, next_states, dones)
