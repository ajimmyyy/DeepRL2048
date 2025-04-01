# utils/logger.py

import logging

class Logger:
    def __init__(self, log_file='training.log'):
        # 設定 logger
        self.logger = logging.getLogger('DQN')
        self.logger.setLevel(logging.INFO)
        
        # 設定日志輸出格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # 設定日誌文件輸出
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 設定終端輸出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, message):
        """記錄訓練過程中的信息"""
        self.logger.info(message)
        print(message)
