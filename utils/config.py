class Config:
    # 模型相關
    STATE_SIZE = 4  # 2048 遊戲棋盤大小 (4x4)
    ACTION_SIZE = 4  # 可能的動作數量 (上、下、左、右)

    # 強化學習相關
    LEARNING_RATE = 0.0005  # 學習率
    GAMMA = 0.99  # 折扣因子
    EPSILON_START = 1.0  # 初始 epsilon
    EPSILON_MIN = 0.01  # 最小 epsilon
    EPSILON_DECAY = 0.995  # epsilon 衰減率
    BATCH_SIZE = 256  # 批次大小
    REPLAY_BUFFER_SIZE = 10000  # 回放緩衝區大小
    TARGET_UPDATE_FREQUENCY = 100  # 更新目標網路的頻率

    # 訓練過程相關
    ENV_NUM = 32  # 環境數量
    MAX_EPISODES = 3001  # 最大訓練回合數
    MAX_STEPS = 500  # 每回合的最大步數

    # 儲存與載入模型
    MODEL_SAVE_PATH = 'checkpoint.pth'  # 儲存模型的路徑
    MODEL_SAVE_INTERVAL = 100  # 儲存模型的間隔
