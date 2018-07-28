import os
import time

import numpy as np

from MazeActions import LEFT, RIGHT, UP, DOWN


class MazeEnv:
    def __init__(self, max_row, max_col, worker, treasure, refresh_interval, obstacles=None):
        self.max_row = max_row
        self.max_col = max_col
        self.init_worker = worker.clone()
        self.worker = worker
        self.treasure = treasure
        self.refresh_interval = refresh_interval

        if obstacles:
            for obstacle in obstacles:
                if treasure.equal(obstacle):
                    raise Exception('The treasure point is conflicted with an obstacle point')
            self.obstacles = obstacles
        else:
            self.obstacles = []

    def move(self, action):
        # print(action)

        if action == LEFT:
            if self.worker.col > 0:
                self.worker.col -= 1
        elif action == RIGHT:
            if self.worker.col < self.max_col - 1:
                self.worker.col += 1
        elif action == UP:
            if self.worker.row > 0:
                self.worker.row -= 1
        elif action == DOWN:
            if self.worker.row < self.max_row - 1:
                self.worker.row += 1
        else:
            raise Exception('Not supported action: {}'.format(action))

        return self.feedback()

    def feedback(self):
        state = self.worker.toString()
        reward = 0
        if self.worker.equal(self.treasure):
            reward = 1
        for obstacle in self.obstacles:
            if self.worker.equal(obstacle):
                reward = -1
        return state, reward

    def reset(self):
        self.worker = self.init_worker.clone()
        return self.worker.toString()

    def display(self):
        os.system('clear')
        arr = np.zeros((self.max_row, self.max_col))
        arr[self.treasure.row][self.treasure.col] = 8
        arr[self.worker.row][self.worker.col] = 1
        for obstacle in self.obstacles:
            arr[obstacle.row][obstacle.col] = 4
        print(arr)
        time.sleep(self.refresh_interval)


class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def equal(self, other_point):
        return self.row == other_point.row and self.col == other_point.col

    def toString(self):
        return "({},{})".format(str(self.row), str(self.col))

    def clone(self):
        return Point(self.row, self.col)
