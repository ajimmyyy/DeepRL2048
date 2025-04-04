import time
import numpy as np
from game_env.parallel_env import ParallelEnv
from agent.dqn_agent import DQNAgent
from utils.reward_writer import RewardWriter
from utils.config import Config
from utils.logger import Logger

# 設定
config = Config()
logger = Logger()

def train():
    # 初始化環境和代理
    envs = ParallelEnv(config.ENV_NUM)
    agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, device_type=config.DEVICE_TYPE)
    reward_writer = RewardWriter()

    for episode in range(config.MAX_EPISODES):
        start_time = time.time()
        states = envs.reset()
        total_rewards = np.zeros(config.ENV_NUM)

        for step in range(config.MAX_STEPS):
            actions = agent.act(np.array([state.flatten() for state in states]))
            new_states, rewards, dones = envs.step(actions)

            for i in range(config.ENV_NUM):
                agent.replay_buffer.add((states[i].flatten(), actions[i], rewards[i], new_states[i].flatten(), dones[i]))

            agent.train()
            total_rewards += rewards
            states = new_states

            if np.all(dones):
                break

        elapsed_time = time.time() - start_time

        # 記錄訓練結果
        logger.log(f"Episode {episode}, Avg Reward: {total_rewards.mean()}, Time: {elapsed_time:.2f} sec")
        reward_writer.append(episode, total_rewards.sum(), total_rewards.mean())

        # 儲存模型
        if episode % config.MODEL_SAVE_INTERVAL == 0:
            agent.save_model(config.MODEL_SAVE_PATH)
            logger.log(f"Model saved at episode {episode}")

        if episode % 100 == 0:
            logger.log(f"Episode {episode}: Total Reward: {total_rewards.sum()}")

    envs.close()
    reward_writer.close()

if __name__ == "__main__":
    train()
