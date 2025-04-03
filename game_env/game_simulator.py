import numpy as np
import random

class Game2048Simulator:
    MERGE_BONUS = 0 # 合併獎勵
    INVALID_MOVE_PENALTY = -4 # 無效移動懲罰
    NEW_TILE_VALUE_PROB = 1 # 新方塊出現的機率
    
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int16)
        self.score = 0
        self.done = False
        self._add_new_tile()
        self._add_new_tile()

    def _add_new_tile(self):
        """在空白格子中隨機加入 2 或 4"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[random.randint(0, len(empty_cells) - 1)]
            self.board[x, y] = 2 if random.random() < self.NEW_TILE_VALUE_PROB else 4

    def _merge(self, row):
        """將 row 靠左壓縮並合併"""
        non_zero = row[row != 0]
        new_row = np.zeros_like(row)
        i, j, score = 0, 0, 0

        while i < len(non_zero):
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                new_row[j] = non_zero[i] * 2
                score += new_row[j]
                i += 2
            else:
                new_row[j] = non_zero[i]
                i += 1
            j += 1

        return new_row, score

    def _move(self, direction):
        """執行 2048 移動 (0:上, 1:下, 2:左, 3:右)"""
        board_copy = self.board.copy()
        if direction in {0, 1}:
            self.board = self.board.T

        if direction in {1, 3}:
            self.board = self.board[:, ::-1]

        score = 0
        for i in range(4):
            self.board[i], row_score = self._merge(self.board[i])
            score += row_score

        if direction in {1, 3}:
            self.board = self.board[:, ::-1]

        if direction in {0, 1}:
            self.board = self.board.T

        changed = not np.array_equal(self.board, board_copy)
        self.score += score

        if changed:
            self._add_new_tile()

        return changed

    def is_game_over(self):
        """檢查遊戲是否結束"""
        if np.any(self.board == 0):
            return False
        for x in range(4):
            for y in range(4):
                if (x < 3 and self.board[x, y] == self.board[x + 1, y]) or \
                   (y < 3 and self.board[x, y] == self.board[x, y + 1]):
                    return False 
        return True

    def step(self, action):
        """執行一個動作 (0:上, 1:下, 2:左, 3:右)，回傳 (新狀態, 獎勵, 是否結束)"""
        prev_score = self.score
        changed = self._move(action)
        reward = (self.score - prev_score) if changed else self.INVALID_MOVE_PENALTY
        self.done = self.is_game_over()

        return self.board.copy(), reward, self.done

    def reset(self):
        """重置遊戲"""
        self.__init__()
        return self.board.copy()