from .game_simulator import Game2048Simulator
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ParallelEnv:
    def __init__(self, num_envs):
        """初始化多個環境"""
        self.num_envs = num_envs
        self.envs = [Game2048Simulator() for _ in range(num_envs)]

    def _step_single(self, env, action):
        """對單一環境執行動作"""
        return env.step(action)
    
    def _reset_single(self, env):
        """對單一環境重置"""
        return env.reset()

    def step(self, actions):
        """對所有環境執行動作，回傳 (新狀態, 獎勳, 是否結束)"""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._step_single, self.envs, actions))

        states, rewards, dones = zip(*results)
        return np.array(states), np.array(rewards), np.array(dones)

    def reset(self):
        """重置所有環境"""
        with ThreadPoolExecutor() as executor:
            states = list(executor.map(self._reset_single, self.envs))
        return np.array(states)
