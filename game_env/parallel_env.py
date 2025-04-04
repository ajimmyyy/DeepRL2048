from .game_simulator import Game2048Simulator
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class Game2048Process(mp.Process):
    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.env = Game2048Simulator()

    def run(self):
        """子進程運行，處理 step 和 reset"""
        while True:
            cmd, data = self.conn.recv()
            if cmd == "step":
                state, reward, done = self.env.step(data)
                self.conn.send((state, reward, done))
            elif cmd == "reset":
                state = self.env.reset()
                self.conn.send(state)
            elif cmd == "close":
                self.conn.close()
                break

class ParallelEnv:
    def __init__(self, num_envs):
        """初始化多個環境"""
        self.num_envs = num_envs
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [Game2048Process(child_conn) for child_conn in self.child_conns]

        for process in self.processes:
            process.start()
            
    def step(self, actions):
        """對所有環境執行 step"""
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))

        results = [conn.recv() for conn in self.parent_conns]
        states, rewards, dones = zip(*results)
        return np.array(states), np.array(rewards), np.array(dones)

    def reset(self):
        """重置所有環境"""
        for conn in self.parent_conns:
            conn.send(("reset", None))
        
        states = [conn.recv() for conn in self.parent_conns]
        return np.array(states)
    
    def close(self):
        """關閉所有環境"""
        for conn in self.parent_conns:
            conn.send(("close", None))

        for process in self.processes:
            process.join()
