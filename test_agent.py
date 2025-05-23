import argparse
import pygame
import numpy as np
import copy
from agent.dqn_agent import DQNAgent
from game_env.game_ui import Game2048UI
from utils.config import Config

config = Config()

class TestAgent:
    def __init__(self, model_path):
        """初始化測試代理"""
        self.agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
        self.agent.load_model(model_path)
        self.agent.epsilon = -1
        self.ui = Game2048UI()

    def is_valid_action(self, action):
        """檢查 action 是否能改變遊戲狀態"""
        temp_env = copy.deepcopy(self.ui.env)
        before_board = temp_env.board.copy()

        temp_env.step(action)
        after_board = temp_env.board

        return not np.array_equal(before_board, after_board)

    def get_game_state(self):
        """獲取遊戲狀態並將其格式化為模型所需的輸入"""
        game_state = self.ui.env.board
        return np.reshape(game_state, (1, 16))

    def play_game(self):
        """讓代理在UI模式下玩遊戲"""
        self.ui.draw_board()
        totol_step = 0
        invalid_action_count = 0

        while not self.ui.env.done:
            game_state = self.get_game_state()
            action = self.agent.act(game_state).item()

            if not self.is_valid_action(action):
                invalid_action_count += 1
                print(f"Invalid action: {action}. Choosing another action.")
                action = np.random.choice([a for a in range(config.ACTION_SIZE) if self.is_valid_action(a)])

            # 在遊戲環境中執行動作
            self.ui.env.step(action)

            # 根據當前遊戲狀態更新UI顯示
            self.ui.draw_board()

            # 暫停一段時間，以便遊戲有時間更新
            pygame.time.wait(200)

            totol_step += 1

        # 遊戲結束時打印結果
        print(f"Game Over! Final Score: {self.ui.env.score}")
        print(f"Total Steps: {totol_step}")
        print(f"Invalid Actions: {invalid_action_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    test_agent = TestAgent(args.model)
    test_agent.play_game()