import torch
import numpy as np
import random
import torch_directml
from collections import deque

class SumTree:
    """SumTree 用於計算優先級抽樣"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0
    
    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
    
    def get_leaf(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]
    
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5, device=None):
        """初始化 PER 緩衝區
        :param buffer_size: 優先級經驗回放的大小
        :param alpha: 優先級因子 (0 - 無優先級, 1 - 完全優先級)
        :param beta: 重要性采样補償初始值
        :param beta_increment: 每次采样後 beta 的增長
        :param epsilon: 避免優先級為零的極小值
        """
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.tree = SumTree(buffer_size)
        self.device = device if device else torch_directml.device()
    
    def add(self, experience):
        # 找出當前最大優先級
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        
        for i in range(batch_size):
            value = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        probs = np.array(priorities) / self.tree.total_priority()
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        return idxs, states, actions, rewards, next_states, dones, weights
    
    def update_priorities(self, idxs, errors):
        priorities = (np.abs(errors) + self.epsilon) ** self.alpha
        sorted_items = sorted(zip(idxs, priorities), key=lambda x: x[0], reverse=True)
        for idx, priority in sorted_items:
            self.tree.update(idx, priority)

    def size(self):
        """返回當前緩衝區的大小"""
        return self.tree.size
    
if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(buffer_size=10)
    
    # 添加經驗
    for i in range(10):
        state = np.array([i, i + 1])
        action = i % 2
        reward = i * 0.1
        next_state = np.array([i + 1, i + 2])
        done = i == 9
        error = random.random()
        buffer.add((state, action, reward, next_state, done), error)
    
    # 取樣
    idxs, states, actions, rewards, next_states, dones, weights = buffer.sample(5)
    print("Sampled indices:", idxs)
    print("States:", states)
    print("Actions:", actions)
    print("Rewards:", rewards)
    print("Next states:", next_states)
    print("Dones:", dones)
    print("Weights:", weights)
