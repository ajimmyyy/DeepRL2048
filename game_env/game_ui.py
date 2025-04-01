import pygame
import numpy as np
from .game_simulator import Game2048Simulator

class Game2048UI:
    ACTIONS = {
        pygame.K_w: 0,     # W 上
        pygame.K_s: 1,     # S 下
        pygame.K_a: 2,     # A 左
        pygame.K_d: 3      # D 右
    }

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("2048")
        self.font = pygame.font.Font(None, 36)
        self.env = Game2048Simulator()
        self.running = True

    def get_tile_color(self, value):
        """根據數字大小返回對應的顏色"""
        if value == 0:
            return (205, 193, 180)  # 默認顏色：淺灰色
        
        log_value = int(np.log2(value))

        # 根據數字大小設定顏色範圍
        if log_value <= 5:
            r = min(255, log_value * 50)
            g = min(255, 255 - log_value * 30)
            b = min(255, 255 - log_value * 20)
        elif log_value <= 10:  # 16, 32, 64, 128, 256, 512
            r = min(255, log_value * 35)
            g = min(255, 255 - log_value * 25)
            b = min(255, 255 - log_value * 15)
        else:  # 1024, 2048, 4096, 8192
            r = min(255, log_value * 20)
            g = min(255, 255 - log_value * 10)
            b = min(255, 255 - log_value * 5)
        
        return (r, g, b)

    def draw_board(self):
        """繪製遊戲畫面"""
        # 填充背景顏色
        self.screen.fill((187, 173, 160))

        for y in range(4):
            for x in range(4):
                value = self.env.board[y, x]

                color = self.get_tile_color(value)

                screen_y = 3 - y
                rect = pygame.Rect(x * 100 + 10, screen_y * 100 + 10, 90, 90)

                pygame.draw.rect(self.screen, color, rect)

                if value:  # 如果該格子有數字（即非零）
                    text_color = (0, 0, 0) if value <= 4 else (255, 255, 255)
                    text = self.font.render(str(value), True, text_color)
                    self.screen.blit(text, (x * 100 + 40, screen_y * 100 + 40))

        pygame.display.flip()


    def run(self):
        """遊戲主迴圈"""
        clock = pygame.time.Clock()
        while self.running:
            clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.ACTIONS:
                        self.env.step(self.ACTIONS[event.key])
                        self.draw_board()

            if self.env.done:
                self.running = False

        pygame.quit()

if __name__ == "__main__":
    game = Game2048UI()
    game.draw_board()
    game.run()
