from .game_simulator import Game2048Simulator
import numpy as np

class ParallelEnv:
    def __init__(self, num_envs):
        """初始化多個環境"""
        self.num_envs = num_envs
        self.envs = [Game2048Simulator() for _ in range(num_envs)]

    def step(self, actions):
        """對所有環境執行動作，回傳 (新狀態, 獎勳, 是否結束)"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones = zip(*results)
        return np.array(states), np.array(rewards), np.array(dones)

    def reset(self):
        """重置所有環境"""
        return np.array([env.reset() for env in self.envs])

# 測試
if __name__ == "__main__":
    num_envs = 4
    parallel_env = ParallelEnv(num_envs)
    states = parallel_env.reset()
    actions = np.random.choice(4, num_envs)
    new_states, rewards, dones = parallel_env.step(actions)
    print(new_states, rewards, dones)
