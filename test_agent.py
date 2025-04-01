import argparse
import pygame
import numpy as np
from agent.dqn_agent import DQNAgent
from game_env.game_ui import Game2048UI

class TestAgent:
    def __init__(self, model_path='dqn_model.pth'):
        """初始化測試代理"""
        self.agent = DQNAgent(state_size=16, action_size=4)
        self.agent.load_model(model_path)
        self.ui = Game2048UI()

    def get_game_state(self):
        """獲取遊戲狀態並將其格式化為模型所需的輸入"""
        game_state = self.ui.env.board
        return np.reshape(game_state, (1, 4, 4, 1))

    def play_game(self):
        """讓代理在UI模式下玩遊戲"""
        self.ui.draw_board()

        while not self.ui.env.done:
            game_state = self.get_game_state()
            action = self.agent.act(game_state)

            # 在遊戲環境中執行動作
            self.ui.env.step(action)

            # 根據當前遊戲狀態更新UI顯示
            self.ui.draw_board()

            # 暫停一段時間，以便遊戲有時間更新
            pygame.time.wait(200)

        # 遊戲結束時打印結果
        print(f"Game Over! Final Score: {self.ui.env.score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    test_agent = TestAgent(args.model)
    test_agent.play_game()