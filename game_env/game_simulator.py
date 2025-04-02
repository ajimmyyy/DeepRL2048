import numpy as np
import random

class Game2048Simulator:
    MERGE_BONUS = 0 # 合併獎勵
    INVALID_MOVE_PENALTY = -4 # 無效移動懲罰
    NEW_TILE_VALUE_PROB = 1 # 新方塊出現的機率
    NEW_TILE_VALUES = [2, 4]
    
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.done = False
        self._add_new_tile()
        self._add_new_tile()

    def _add_new_tile(self):
        """在空白格子中隨機加入 2 或 4"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = self.NEW_TILE_VALUES[0] if random.random() < self.NEW_TILE_VALUE_PROB else self.NEW_TILE_VALUES[1]

    def _compress(self, row):
        """將 row 靠左壓縮，例如 [2, 0, 2, 4] 變成 [2, 2, 4, 0]"""
        row = [i for i in row if i != 0]  # 移除 0
        row += [0] * (4 - len(row))       # 填充 0
        return row

    def _merge(self, row):
        """合併相同數字，例如 [2, 2, 4, 0] 變成 [4, 4, 0, 0]"""
        for i in range(3):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0
        return self._compress(row)

    def _move(self, direction):
        """執行 2048 移動 (0:上, 1:下, 2:左, 3:右)"""
        rotated = self.board

        if direction == 0:  # 上
            rotated = rotated.T[:, ::-1]
        elif direction == 1:  # 下
            rotated = rotated.T
        elif direction == 2:  # 左
            rotated = rotated
        elif direction == 3:  # 右
            rotated = rotated[:, ::-1]

        moved_board = np.array([self._merge(self._compress(row)) for row in rotated])

        # 還原旋轉方向
        if direction == 0:  # 上
            moved_board = moved_board[:, ::-1].T
        elif direction == 1:  # 下
            moved_board = moved_board.T
        elif direction == 2:  # 左
            moved_board = moved_board
        elif direction == 3:  # 右
            moved_board = moved_board[:, ::-1]

        changed = not np.array_equal(self.board, moved_board)
        self.board = moved_board

        if changed:
            self._add_new_tile()

        return changed

    def is_game_over(self):
        """檢查遊戲是否結束"""
        if np.any(self.board == 0):  # 還有空格
            return False
        for x in range(4):
            for y in range(4):
                if (x < 3 and self.board[x, y] == self.board[x + 1, y]) or \
                   (y < 3 and self.board[x, y] == self.board[x, y + 1]):
                    return False  # 存在可合併的數字
        return True

    def step(self, action):
        """執行一個動作 (0:上, 1:下, 2:左, 3:右)，回傳 (新狀態, 獎勵, 是否結束)"""
        prev_score = self.score
        prev_empty_cells = np.sum(self.board == 0)

        changed = self._move(action)

        new_empty_cells = np.sum(self.board == 0)
        merge_bonus = (prev_empty_cells - new_empty_cells) * self.MERGE_BONUS
        
        reward = self.score - prev_score + merge_bonus if changed else self.INVALID_MOVE_PENALTY

        self.done = self.is_game_over()
        return self.board.copy(), reward, self.done

    def reset(self):
        """重置遊戲"""
        self.__init__()
        return self.board.copy()