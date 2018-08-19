
import numpy as np
import tensorflow as tf
import time
import sys

import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(self,n_actions,n_features,learning_rate = 0.01,
                 reward_decay = 0.9,e_greedy = 0.9,replace_target_iter = 300,
                 memory_size = 500,batch_size = 32,e_greedy_increment = None,
                 output_graph = False):
        self.n_actions = n_actions
        self.n_features = n_features                        # 特征数
        self.lr = learning_rate
        self.gamma = reward_decay                           # 奖励衰退率，可以理解为梯度学习率下降，损失下降这些。
        self.epsilon_max = e_greedy                         # 贪心得到最大奖励。
        self.replace_target_iter = replace_target_iter      # 代替目标的迭代次数
        self.memory_size = memory_size                      # 记忆池容量,理解为样本容量
        self.batch_size = batch_size                        # 一次batch数
        self.epsilon_increment = e_greedy_increment         # 奖励增长概率/值。
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 贪心增长的结果，也就是奖励值。如果没有，取最大奖励。

        self.learn_step_counter = 0                         # 学习次数
        '''
        feature是特征数，在这里是坐标值，也就是每次记忆都包含了该样本，横坐标是样本数，纵坐标是特征数。
        '''
        self.memory = np.zeros(self.memory_size,n_features * 2 +2)

    def _build_net(self):
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name = 's')           # 当前状态    要求是样本(batch),特征
        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name = 's_')         # 下一个状态
        self.r = tf.placeholder(tf.float32,[None,],name = 'r')                          # 每个样本的奖励值
        self.a = tf.placeholder(tf.int32,[None,],name = 'a')                            # 行动数

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        '''
        Q-learning的网络是俩种，一个目标，一个评估。
        '''
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer = w_initializer,
                                 bias_initializer = b_initializer,name = 'e1')
            self.q_eval = tf.layers.dense(e1,self.n_actions,kernel_initializer = w_initializer,
                                   bias_constraint = b_initializer,name = 'q')
        '''
        目标网络
        '''
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_,20,tf.nn.relu,kernel_initializer = w_initializer,
                                 bias_constraint = b_initializer,name = 't1')
            self.q_next = tf.layers.dense(t1,self.n_actions,kernel_initializer = w_initializer,
                                          bias_initializer = b_initializer,name = 't2')

        '''
        我们的最大奖励值
        axis：要减少的尺寸。如果None（默认值），则减少所有尺寸。必须在范围内 [-rank(input_tensor), rank(input_tensor))。
        reduce_max: 计算张量中的最大值，axis = 1 也就是每一列的最大值。
        tf.stop_gradient 简单说就是终止图中正计算的梯度值，而我们的目标值正是有损失值的梯度计算着，所以当我们
        得到目标值时，我们会终止它。
        '''
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            # 一个节点被 stop之后，这个节点上的梯度，就无法再向前BP了
            self.q_target = tf.stop_gradient(q_target)

        '''
        获取评估的Q值,合并矩阵
        '''
        with tf.variable_scope('q_eval'):
            # tf.stack
            # a = tf.constant([1,2,3])
            # b = tf.constant([4,5,6])
            # c = tf.stack([a,b],axis=1)
            # [[1 4]
            #  [2 5]
            # [3 6]]
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0],dtype = tf.int32),self.a],axis = 1)
            # tf.gather_nd解析如下：
            # 用indices从张量params得到新张量
            # indices = [[0, 0], [1, 1]]
            # params = [['a', 'b'], ['c', 'd']]
            # output = ['a', 'd']
            # 这里self.q_eval是batch * action_number,a_indices是batch * 1，也就是说选择当前估计每个动作的Q值
            self.q_eval_wrt_a = tf.gather_nd(paramse = self.q_eval,indices = a_indices)

        '''
        以元素方式返回（x-y）（x-y）。
        '''
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        '''
        注意，在该算法的密集实现中，即使梯度为零，也将更新变量及其相应的累加器（动量，梯度移动平均值，平方梯度移动平均值）
        （即累加器将衰减，将应用动量）。稀疏实现（当梯度是一个IndexedSlices对象时使用，通常是因为tf.gather正向传递中的嵌入查找）
        将不会更新变量切片或它们的累加器，
        除非这些切片在前向传递中使用（也没有“最终”校正考虑这些省略的更新）。这样可以更有效地更新大型嵌入查找表
        （大多数切片不会在特定的图执行中访问），但与已发布的算法不同。
        '''
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)\

    '''
    更新记忆池。
    '''
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # hstack:Stack arrays in sequence horizontally
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    '''
    选择action
    '''
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.random() < self.epsilon:
            action_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    '''
    开始学习
    '''
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:]
            }
        )

        self.cost_his.append(cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

'''
数据集类
'''
class Maze():
    def __init__(self,a):
        self.a = a

def run_maze():
    step = 0
    for episode in range(300):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_,reward,done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    #env.after(100, run_maze)
    #env.mainloop()
    RL.plot_cost()
