# DeepRL2048

Deep Reinforcement Learning for 2048 Game

📌 專案簡介

本專案使用 深度強化學習 (Deep Reinforcement Learning, DRL) 來訓練 AI 玩 2048 遊戲。主要採用 Deep Q-Network (DQN) 來學習最佳策略，並使用 PyTorch 進行建模與訓練。

📥 安裝

1️⃣ 安裝 Python (環境為Python 3.13)

2️⃣ 建立虛擬環境

```bach
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows 
```

3️⃣ 安裝必要套件

```bach
pip install -r requirements.txt
```

🎮 運行與訓練

1️⃣ 訓練模型

運行 train_dqn.py 開始訓練 DQN

```bach
python train_dqn.py
```

接續訓練 DQN

```bach
python train_dqn.py --model checkpoints/checkpoint.pth
```

運行 reward_dashboard.py 查看訓練過程

```bach
streamlit run ui/reward_dashboard.py
```

2️⃣ 測試訓練結果

```bach
python test_agent.py --model checkpoints/checkpoint.pth
```

📝 調整

utils/config.py 下能調整模型訓練參數  
models 下能實現不同模型細節

📝 TODO
