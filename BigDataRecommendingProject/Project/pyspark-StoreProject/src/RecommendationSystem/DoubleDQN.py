

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 决定是否要使用double - q -network
        self.double_q = double_q

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    '''
    这里我们的双网络结构都简单的采用简单的全链接神经网络，
    包含一个隐藏层。这里我们得到的输出是一个向量，表示该状态才取每个动作可以获得的Q值
    '''
    def _build_net(self):
        def build_layers(s, c_name, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(name='w1', shape=[self.n_features, n_l1], initializer=w_initializer,
                                     collections=c_name)
                b1 = tf.get_variable(name='b1', shape=[1, n_l1], initializer=b_initializer, collections=c_name)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(name='w2', shape=[n_l1, self.n_actions], initializer=w_initializer,
                                     collections=c_name)
                b2 = tf.get_variable(name='b2', shape=[1, self.n_actions], initializer=b_initializer,
                                     collections=c_name)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------------input---------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q-target')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 20
            w_initializer = tf.random_normal_initializer(0, 0.3)
            b_initializer = tf.constant_initializer(0.1)
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    '''
    选择action
    我们仍然使用的是e-greedy的选择动作策略，即以e的概率选择随机动作，
    以1-e的概率通过贪心算法选择能得到最多奖励的动作a。
    '''
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if np.random.random() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):

        '''
        更新target-net
        这里，每个一定的步数，我们就更新target-net中的参数：
        :return:
        '''
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        '''
        选择数据batch
        我们从经验池中选择我们训练要使用的数据。
        '''
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        '''
        更新网络参数
        根据Double DQN的做法，我们需要用两个网络的来计算我们的q-target值，
        同时通过最小化损失来更新网络参数。这里的做法是，根据 eval-net 的值来选择动作，然后根据target-net的值来计算Q值。
        q_next是根据经验池中下一时刻状态输入到target-net计算得到的q值，
        而q_eval4next是根据经验池中下一时刻状态s'输入到eval-net计算得到的q值，这个q值主要用来选择动作。
        '''
        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                                                       self.s: batch_memory[:, -self.n_features:]})

        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :self.n_features]})

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        '''
        接下来，我们就要来选择动作并计算该动作的q值了,如果是double dqn的话，
        我们是根据刚刚计算的q_eval4next来选择动作，然后根据q_next来得到q值的。而原始的dqn直接通过最大的q_next来得到q值：
        '''
        if self.double_q:
            max_act4next = np.argmax(q_eval4next,
                                     axis=1)  # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        '''
        那么我们的q-target值就可以计算得到了：
        '''
        q_target = q_eval.copy()
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        
        '''
        有了q-target值，我们就可以结合eval-net计算的q-eval值来更新网络参数了：
        '''
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
