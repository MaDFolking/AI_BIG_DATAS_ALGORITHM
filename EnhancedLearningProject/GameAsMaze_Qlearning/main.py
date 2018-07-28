from MazeActions import LEFT, RIGHT, UP, DOWN
from MazeEnv import MazeEnv, Point
from QLearningAgent import QLearningAgent

"""
使用强化学习走迷宫,由于刚开始的状态是随机的,所以当迷宫面积较大的时候,可能会导致训练时间太长,可以尝试利用 A* 算法原理,
帮助更快找到treasure
迷宫展示字典:
    1: worker
    4: obstacle
    8: treasure
reward字典:
    -1: obstacle
     1: treasure
经测试,跑完 30 个 episode,约耗时 30s
"""

if __name__ == '__main__':
    refresh_interval = 0.05  # 该参数用于定时显示迷宫情况

    episode = 30  # 训练多少回合

    epsilon = 0.9  # 使用历史经验的概率, 若值为0.9,则有 90% 的情况下,会根据历史经验选择 action, 10% 的情况下,随机选择 action
    learning_rate = 0.01  # 根据公式可知,该值越大,则旧训练数据被保留的就越少
    discount_factor = 0.9  #

    max_row = 4
    max_col = 4
    actions = [LEFT, RIGHT, UP, DOWN]
    worker = Point(0, 0)
    treasure = Point(2, 2)
    obstacles = [
        Point(1, 2),
        Point(2, 1),
    ]

    env = MazeEnv(
        max_row=max_row,
        max_col=max_col,
        worker=worker,
        treasure=treasure,
        obstacles=obstacles,
        refresh_interval=refresh_interval
    )
    agent = QLearningAgent(
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        actions=actions
    )
    successful_step_counter_arr = []
    failed_step_counter_arr = []

    env.display()

    for eps in range(1, episode + 1):

        cur_state = env.reset()
        step_counter = 0

        while True:
            step_counter += 1

            env.display()

            action = agent.choose_action(cur_state)

            next_state, reward = env.move(action)

            agent.learn(
                cur_state=cur_state,
                action=action,
                reward=reward,
                next_state=next_state
            )

            cur_state = next_state

            if reward != 0:
                break

        if reward > 0:
            successful_step_counter_arr.append(step_counter)
        elif reward < 0:
            failed_step_counter_arr.append(step_counter)

        print(
            'total episode: {}\n'
            'current episode: {}\n'
            'reward: {}\nsteps: {}\n'
            'successful steps record: {}\n'
            'failed steps record: {}'
                .format(
                episode,
                eps,
                reward,
                step_counter,
                successful_step_counter_arr,
                failed_step_counter_arr
            )
        )
        input('press enter to next episode...')

        # print(agent.q_table)
        # print('successful steps record: {}'.format(succeed_step_counter_arr))
