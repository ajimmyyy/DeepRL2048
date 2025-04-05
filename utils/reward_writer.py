import json
import os

class RewardWriter:
    def __init__(self, filepath="reward_data.json"):
        self.filepath = filepath
        self.data = {"episode": [], "total_reward": [], "avg_reward": [], "elapsed_time": []}

        # 初始化或清除舊資料
        with open(self.filepath, "w") as f:
            json.dump(self.data, f)

    def append(self, episode, total_reward, avg_reward, elapsed_time):
        self.data["episode"].append(episode)
        self.data["total_reward"].append(total_reward)
        self.data["avg_reward"].append(avg_reward)
        self.data["elapsed_time"].append(elapsed_time)

        with open(self.filepath, "w") as f:
            json.dump(self.data, f)

    def close(self):
        pass