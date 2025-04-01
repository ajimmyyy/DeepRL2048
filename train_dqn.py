import numpy as np
from game_env.parallel_env import ParallelEnv
from agent.dqn_agent import DQNAgent
from utils.config import Config
from utils.logger import Logger

# 設定
config = Config()
logger = Logger()

# 初始化環境和代理
envs = ParallelEnv(config.ENV_NUM)
agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)

for episode in range(config.MAX_EPISODES):
    states = envs.reset()
    total_rewards = np.zeros(config.ENV_NUM)

    for step in range(config.MAX_STEPS):
        actions = [agent.act(state.flatten()) for state in states]
        new_states, rewards, dones = envs.step(actions)

        for i in range(config.ENV_NUM):
            agent.replay_buffer.add((states[i].flatten(), actions[i], rewards[i], new_states[i].flatten(), dones[i]))

        loss = agent.train()
        total_rewards += rewards
        states = new_states

        if np.all(dones):
            break

    # 記錄訓練結果
    logger.log(f"Episode {episode}, Avg Reward: {total_rewards.mean()}, Loss: {loss}")

    # 儲存模型
    if episode % config.MODEL_SAVE_INTERVAL == 0:
        agent.save_model(config.MODEL_SAVE_PATH)
        logger.log(f"Model saved at episode {episode}")

    if episode % 100 == 0:
        logger.log(f"Episode {episode}: Total Reward: {total_rewards.sum()}")
