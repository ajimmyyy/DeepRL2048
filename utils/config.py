class Config:
    # 模型相關
    STATE_SIZE = 4  # 2048 遊戲棋盤大小 (4x4)
    ACTION_SIZE = 4  # 可能的動作數量 (上、下、左、右)

    # PRIORITIZED REPLAY BUFFER 相關
    ALPHA = 0.6  # 優先級因子 (0 - 無優先級, 1 - 完全優先級)
    BETA = 0.4  # 重要性采樣補償初始值
    BETA_INCREMENT = 0.001  # 每次采样後 beta 的增長
    PER_EPSILON = 1e-5  # 避免優先級為零的極小值

    # 強化學習相關
    LEARNING_RATE = 0.0001  # 學習率
    GAMMA = 0.99  # 折扣因子
    EPSILON_START = 1.0  # 初始 epsilon
    EPSILON_MIN = 0.02  # 最小 epsilon
    EPSILON_DECAY = 0.995  # epsilon 衰減率
    BATCH_SIZE = 1024  # 批次大小
    REPLAY_BUFFER_SIZE = 100000  # 回放緩衝區大小
    TARGET_UPDATE_FREQUENCY = 2500  # 更新目標網路的頻率

    # 訓練過程相關
    DEVICE_TYPE = 'directml' # 使用的設備類型 ('cpu', 'cuda', 'directml' )
    ENV_NUM = 32  # 環境數量
    MAX_EPISODES = 10001  # 最大訓練回合數
    MAX_STEPS = 1000  # 每回合的最大步數

    # 儲存與載入模型
    MODEL_SAVE_PATH = 'checkpoints/checkpoint.pth'  # 儲存模型的路徑
    MODEL_SAVE_INTERVAL = 100  # 儲存模型的間隔
    REWARD_LOG_PATH = 'runs/train_logs'  # 獎勵日誌的路徑

    def get_summary(self):
        """列印配置參數"""
        return {
            'STATE_SIZE': Config.STATE_SIZE,
            'ACTION_SIZE': Config.ACTION_SIZE,
            'ALPHA': Config.ALPHA,
            'BETA': Config.BETA,
            'BETA_INCREMENT': Config.BETA_INCREMENT,
            'PER_EPSILON': Config.PER_EPSILON,
            'LEARNING_RATE': Config.LEARNING_RATE,
            'GAMMA': Config.GAMMA,
            'EPSILON_START': Config.EPSILON_START,
            'EPSILON_MIN': Config.EPSILON_MIN,
            'EPSILON_DECAY': Config.EPSILON_DECAY,
            'BATCH_SIZE': Config.BATCH_SIZE,
            'REPLAY_BUFFER_SIZE': Config.REPLAY_BUFFER_SIZE,
            'TARGET_UPDATE_FREQUENCY': Config.TARGET_UPDATE_FREQUENCY,
            'ENV_NUM': Config.ENV_NUM,
            'MAX_EPISODES': Config.MAX_EPISODES,
            'MAX_STEPS': Config.MAX_STEPS,
        }