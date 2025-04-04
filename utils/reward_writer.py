import json
import os

class RewardWriter:
    def __init__(self, filepath="reward_data.json"):
        self.filepath = filepath
        self.data = {"episode": [], "total_reward": [], "avg_reward": []}

        # 初始化或清除舊資料
        with open(self.filepath, "w") as f:
            json.dump(self.data, f)

    def append(self, episode, total_reward, avg_reward):
        self.data["episode"].append(episode)
        self.data["total_reward"].append(total_reward)
        self.data["avg_reward"].append(avg_reward)

        with open(self.filepath, "w") as f:
            json.dump(self.data, f)

    def close(self):
        pass