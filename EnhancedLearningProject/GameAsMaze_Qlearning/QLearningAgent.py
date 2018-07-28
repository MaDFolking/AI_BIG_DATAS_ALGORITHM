import random
import sys

import numpy as np
import pandas as pd


class QLearningAgent:
    def __init__(self, epsilon, learning_rate, discount_factor, actions):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = pd.DataFrame(columns=actions)

    def choose_action(self, state):
        self.create_state_if_not_exists(state)

        if np.random.uniform() < self.epsilon:
            actions = self.q_table.loc[state, :]
            action = self.choose_best_action(actions)
        else:
            action = np.random.choice(self.actions)

        return action

    def build_index(self, max_row, max_col):
        from MazeEnv import Point
        index = []
        for row in range(max_row):
            for col in range(max_col):
                index.append(Point(row, col).toString())
        return index

    def learn(self, cur_state, action, reward, next_state):
        self.create_state_if_not_exists(next_state)
        q_predict = self.q_table.loc[cur_state, action]
        if self.not_finished(reward):
            q_reality = reward + self.discount_factor * self.q_table.loc[next_state, :].max()
        else:  # 代表碰到 obstacle or tresure
            q_reality = reward
        self.q_table.loc[cur_state, action] += self.learning_rate * (q_reality - q_predict)

    def create_state_if_not_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    data=[0] * len(self.actions),
                    index=self.actions,
                    name=state
                )
            )

    def not_finished(self, reward):
        return reward == 0

    def choose_best_action(self, actions):
        q_table_cols = self.q_table.columns
        max_action_value = -sys.maxsize
        max_action_value_list = []

        for idx in range(len(q_table_cols)):
            action_value = actions[idx]
            q_tabl_col = q_table_cols[idx]

            if action_value > max_action_value:
                max_action_value = action_value
                max_action_value_list = [q_tabl_col]
            elif action_value == max_action_value:
                max_action_value_list.append(q_tabl_col)
            else:
                continue

        if len(max_action_value_list) > 1:
            random_action_index = random.randint(0, len(max_action_value_list) - 1)
            best_action = max_action_value_list[random_action_index]
        else:
            best_action = max_action_value_list[0]

        return best_action
