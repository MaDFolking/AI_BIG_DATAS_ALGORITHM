

import os
import time
import logging
import numpy as np
import tensorflow as tf
import tf_utils as tf_util

from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.base import RegressorMixin, BaseEstimator
from util_model import log_time, memory_info,binarize, get_mean_percentiles, get_percentiles

os.environ['OMP_NUM_THREADS'] = '1'
log_time = partial(log_time, name='tf_sparse')

'''
该模块是tensorflow实现神经网络的模块，我顺便复习下tensorflow知识。
'''

'''
这里我一一复习下我们使用的 tensor 知识：

tf.Variable: 构造函数需要为变量,也就是初始化tensor需要的各种变量。可以理解为开辟一个适合tensor的空间变量，调用时
自动转化为tensor需要的变量。而get_variable是直接初始化原有或创建一个新的tensor变量，是一个以存在的变量。

1.tf.get_variable:tensor 中用于初始化原有变量,或者创建一个新变量。
(1)name:新变量/现有变量名字。如果重复创建俩个该变量，指向同一个name的内存地址。
(2)initializer:初始化操作，如果是tensor类型，如果不设定为False就必须指定shape

2.tf.global_variables_initializer()：我们的变量为初始化全局变量，这里配合get_variable意思是初始化全局变量w

3.tf.sparse_tensor_dense_matmul:是个矩阵乘法，将稀疏矩阵转化为密集矩阵。但有个前提，第二个参数必须是密集矩阵，第一个
参数是稀疏矩阵。也就是第二个参数是密集矩阵乘以第一个参数是稀疏矩阵。

4.tf.placeholder: 给tensor向量初始化占位符。相当于开辟一个内存空间。
注意点：如果进行评估，此张量将产生错误。其值必须使用被馈送feed_dict可选参数
Session.run()， Tensor.eval()或Operation.run()

5.tf.SparseTensor:将三种向量合并成一个稀疏张量。
indices:表示向量的位置，values是值，dense_shape是维度。
SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
代表密集的张量
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]

6.tf.squeeze 移除里面为1的数值。未来版本可能被删除。

7.tf.nn.l2_loss： output = sum(t ** 2) / 2 注意返回值还除以2，估计是防止过大。
  这个t必须维度跟我们里面矩阵一样，通常用初始化w的维度矩阵当做t.

8.TensorFlow计算，表示为数据流图。
A Graph包含一组 tf.Operation对象，表示计算单位; 和 tf.Tensor对象，表示在操作之间流动的数据单位。
Graph始终注册默认值，并可通过调用访问 tf.get_default_graph。要将操作添加到默认图形，只需调用定义新的函数之一Operation：
c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()
另一个典型用法涉及 tf.Graph.as_default 上下文管理器，它会覆盖上下文生命周期的当前默认图：
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g
重要说明：此类对于图构建不是线程安全的。应从单个线程创建所有操作，或者必须提供外部同步。除非另有说明，否则所有方法都不是线程安全的。

9.
tf.glorot_normal_initializer()
Glorot普通初始化器，也称为Xavier普通初始化器。
它从以0为中心的截断正态分布中抽取样本，stddev = sqrt(2 / (fan_in + fan_out)) 其中fan_in，
权重张量中的输入单位fan_out数是，并且是权重张量中的输出单位数。

'''


'''
返回原对象,用于返回函数return时，以函数形式返回的状态,用于我们的激活函数包裹着我们隐层，这里没有卷积层了，
直接是隐层->激活函数->隐层。
'''
def identity(x):
    return x

'''
初始化稀疏的线性参数w和b,注意b设置为列向量，返回权重系数w。这个权重是我们未来损失值的权重.
因为tf.nn.l2_loss中，里面输出是 = sum(t**2) /2 要求t必须与矩阵数相同，所以返回w以此来作用。
tf.glorot_normal_initializer():注意，这里用这类正态分布的普通初始化，stddev = sqrt(2 / (fan_in + fan_out))
其中fan_in，权重张量中的输入单位fan_out数是，并且是权重张量中的输出单位数。
'''
def sparse_linear(xs,shape,name:str,actfunc = identity):
    assert len(shape) == 2
    w = tf.get_variable(name = name,initializer = tf.glorot_normal_initializer(),shape = shape)
    b = tf.zeros(shape = shape[1])
    return actfunc(tf.sparse_tensor_dense_matmul(xs,w) + b),w

'''
初始化线性参数
'''
def linear(xs,shape,name:str,actfunc = identity):
    assert len(shape) == 2
    w = tf.get_variable(name = name,initializer = tf.glorot_normal_initializer(),shape = shape)
    b = tf.zeros(shape = shape[1])
    return actfunc(tf.matmul(xs,w)+b)

'''
我们的基线，也就是衡量指标
'''
def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

'''
修改relu激活函数，防止过拟合，根据我们的数据而调参数。
'''
def prelu(_x):
    alphas = tf.get_variable('prelu_alpha_{}'.format(_x.get_shape()[-1]), _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

'''
设定sigmod函数
'''
def swish(x, name='swish'):
    """The Swish function, see `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`_.
    Parameters
    ----------
    x : a tensor input
        input(s)
    Returns
    --------
    A `Tensor` with the same type as `x`.
    """
    with tf.name_scope(name) as scope:
        x = tf.nn.sigmoid(x) * x
    return x

'''
初始化稀疏矩阵,并进行二值化处理来优化网络。实现模型压缩。

二值化神经网络，是指在浮点型神经网络的基础上，将其权重矩阵中权重值和各个激活函数值同时进行二值化得到的神经网络。
二值化神经网络具有很好的特性，具体体现在：
(1)通过将权重矩阵二值化，一个权重值只占用一个比特，相比于单精度浮点型权重矩阵，网络模型的内存消耗理论上能减少32倍，
因此二值化神经网络在模型压缩上具有很大的优势。
(2)当权重值和激活函数值同时进行二值化之后，原来32个浮点型数的乘加运算，可以通过一次异或运算(xnor)和一次popcnt运算解决，
在模型加速上具有很大的潜力。
总结：将权重值和激活函数值二值化到-1 和 +1 来实现，理论缩小32倍。 因为原来的是float32双精度，占据32个字节。

对于每个数据集的4个模型中的2个，我们通过将所有非零值设置为1来在训练和预测期间对输入数据进行二值化。
这就像使用二进制CountVectorizer而不是TFIDF免费获得额外数据集。也就是将各种数字，化为1，-1俩种 b 字节的处理，大大降低了内存消耗。
'''
class SparseMatrix:
    '''
    indices:我们向量的位置。因为我们处理的文本是第二个位置，所以这里设置为[None,2],代表着处理第二个位置。
    values: 我们输入的向量值，必须是float32
    dense_shape: 我们的维度，为2
    tensor:我们的tensor单位，也就是以上之和。
    我们下面feed_dict将indices进行矩阵列转换，这样，取第二个数就是我们的文本内容位置。
    '''
    def __init__(self):
        self.indices = tf.placeholder(tf.int64,shape = [None,2])
        self.values = tf.placeholder(tf.float32,shape = [None])
        self.dense_shape = tf.placeholder(tf.int64,shape = [2])
        self.tensor = tf.SparseTensor(
            indices = self.indices,
            values = self.values,
            dense_shape = self.dense_shape,
        )

    '''
    设定是否进行二值化处理,不进行二值化，我们就将非零值都设置为1，同样加快速度。
    np.stack:里面列表大小必须相同，默认是axis = 0,也就是按照行向量转换。
    如下：[[1, 2, 3], [4, 5, 6]]
        增加一维，新维度下标为0，也就是把每行都放到一个列表中，从上到下排开。
        [[1 2 3]
        [4 5 6]]
    如果是列:axis = 1 也就是把每列都放到一个列表中，从上到下排开。
        [[1 4]
         [2 5]  这个就是我们文本内容位置。
        [3 6]]
    我们这里设置为转置，其实就是按列排。
    np.hstack() 水平堆叠：
    a=[[1],[2],[3]]
    b=[[1],[2],[3]]
    c=[[1],[2],[3]]
    d=[[1],[2],[3]]
    print(np.hstack((a,b,c,d)))

    输出：
    [[1 1 1 1]
    [2 2 2 2]
    [3 3 3 3]]
    如果是vstack((a,b,c,d))
    [[1]
     [2]
     [3]
     [1]
     [2]
     [3]
     [1]
     [2]
     [3]
     [1]
     [2]
     [3]]
    np.ones_like(x) 将x矩阵数都变成1.
    '''
    def feed_dict(self,X,isBinary = False):
        coo = X.tocoo()  #tocoo:返回该矩阵的稀疏压缩副本，为了节约内存使用。直接调用里面data取值即可。我们这里用来二值化。
        return{
            self.indices:np.stack([coo.row,coo.col]).T,
            self.values:coo.data if not isBinary else np.ones_like(coo.data),
            self.dense_shape:np.array(X.shape)
        }

'''
初始化MLP神经网络，这是我们的父类函数，我们之后的神经网络，都继承它。
'''
class Regression(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=(16, 16),
                 learning_rate=1e-2, lr_decay=0.75,
                 n_epoch=4, batch_size=2 ** 10, bs_decay=2,
                 decay_epochs=2,
                 reg_l2=1e-5, reg_l1=0.0, use_target_scaling=True, n_bins=64, seed=0,
                 actfunc=tf.nn.relu, binary_X=False):
        self.n_hidden = n_hidden                         # 隐层
        self.reg_l2 = reg_l2                             # L2正则系数，相当于lambda
        self.reg_l1 = reg_l1                             # L1正则系数，相当于alpha
        self.learning_rate = learning_rate               # 学习率
        self.lr_decay = lr_decay                         # 线性衰退率
        self.bs_decay = bs_decay                         # batch改变率，如果epoch小于2时，让我们的迭代batch*2，从而下次迭代时batch增加，这是为了更好地学习。
        self.decay_epochs = decay_epochs                 # epoch的阈值，我们初始设置为2，意思是小于2时，就开始进行线性和非线性衰退操作，让他乘以0.75进行逐渐衰退，防止过拟合。
        self.n_epoch = n_epoch                           # epoch数量。
        self.batch_size = batch_size                     # 一次迭代个数
        self.use_target_scaling = use_target_scaling     # 数据是否标准化
        if self.use_target_scaling:
            self.target_scaler = StandardScaler()
        self.n_bins = n_bins                             # 输入个数
        self.seed = seed                                 # 种子数量
        self.actfunc = actfunc                           # 选择哪个激活函数，默认为relu
        self.binary_X = binary_X                         # 是否进行二值化处理
        self.is_fitted = False                           # 是否拟合数据，默认False,调用fit函数，则自动改为True

    '''
    初始化模型参数
    '''
    def build_model(self,n_features:int):
        self.predict = tf.placeholder(tf.float32,shape = [None])  #预测值
        self.build_hidden(n_features)
        '''
        #输出结果为横坐标是最后一个隐层，纵坐标就是output值1.我们每次横坐标都是前一个隐层，纵坐标是下个隐层。
        我们用squeeze删除里面数值为1的数值，因为前面二值化或者都设置为1的考量。
        '''
        self.output = tf.squeeze(linear(self._hidden,[self.n_hidden[-1],1],'l_last'),axis = 1)
        '''
        定义损失，mse
        '''
        self.loss = tf.losses.mean_squared_error(self.output,self.predict)
        self.add_regularization()

    '''
    初始化正常矩阵和稀疏矩阵的隐层。为了方便全连接，维度为我们的特征数和第一个隐层.
    我们设置了三种状态，用来初始化开始的稀疏矩阵
    '''
    def build_hidden(self,n_features:int):
        hidden,self.w1 = sparse_linear(
            self._xs.tensor,
            [n_features,self.n_hidden[0]],
            'l1',
            self.actfunc
        )
        '''
        如果有3个隐层，根据MLP性质，第一个维度为第一个隐层，第二个维度为第二个隐层，此时名字设置为l2
        '''
        if len(self.n_hidden) == 3:
            hidden = linear(hidden,[self.n_hidden[0],self.n_hidden[1]],'l2',self.actfunc)
        self._hidden = linear(hidden,[self.n_hidden[-2],self.n_hidden[-1]],'l_hidden',self.actfunc)
    '''
    我们的关键点，设置不同损失,我们的损失前面系数都有l2,l1正则系数。
    '''
    def add_regularization(self):
        if self.reg_l2:
            self.loss += self.reg_l2 * tf.nn.l2_loss(self.w1)
        if self.reg_l1:
            self.loss += self.reg_l1 * tf.reduce_sum(tf.abs(self.w1))  #损失和，未来被弃用。也就是我们默认为l1参数。

    '''
    标准化后拟合数据,必须reshape成-1,1 也就是列向量。
    '''
    def _scaler_fit(self, y):
        self.target_scaler.fit(y.reshape(-1, 1))

    '''
    标准化拟合后，数据transform操作，因为是列向量。取所有行，我们的目的是取里面所有文本数据
    '''
    def _scale_target(self, y):
        y = np.array(y)
        if self.use_target_scaling:
            return self.target_scaler.transform(y.reshape(-1, 1))[:, 0]
        return y

    '''
    二值化
    '''
    def _x_feed_dict(self, X):
        return self._xs.feed_dict(X, self.binary_X)

    '''
    预测结果,预测结果时，一定要转化为原来数据，inverse_transform就是这个用处，然后reshape[-1,1][:,0] 取所有横坐标向量。
    这个在MLP/CNN中以前也学过。一定要切记。
    '''
    def _invert_target(self, y):
        y = np.array(y)
        if self.use_target_scaling:
            return self.target_scaler.inverse_transform(y.reshape(-1, 1))[:, 0]
        return y

    '''
    拟合操作.
    X是我们的原始数据，也即是shape[0] = 样本数，shape[1]为特征数
    注意异常: TypeError: 'Tensor' object is not callable
    解析: 函数和变量名相同，导致了再次调用函数时的问题
    '''
    def fit(self,X,y,X_valid = None,y_valid = None,use_gpu = True,verbose = True,to_predict = None):
        self.is_fitted = True
        if self.use_target_scaling:
            self._scaler_fit(y)
            y = self._scale_target(y)
        if y_valid is not None:
            y_valid_scaled = self._scale_target(y_valid)
        n_features = X.shape[1]
        self._graph = tf.Graph()                #开始先构建计算流图
        '''
        设置环境: CPU/GPU 如果是CPU:就使用4核。GPU，就使用我们上面use_gpu的数量。也就是0/1
        allow_soft_placement：
         //是否允许软放置。如果allow_soft_placement为true，
         //如果将操作放在CPU上
        // 1. OP没有GPU实现
        // 要么
        // 2.没有已知或注册的GPU设备
        // 要么
        // 3.需要与来自CPU的reftype输入共存。
        其他都是线程参数。
        ConfigProto 具体看: https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/protobuf/config.proto
        '''
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,  # 对于单个op可执行并行化，以后可以设定
            inter_op_parallelism_threads=1,  # 执行阻塞操作的节点在池中排队,就是java的阻塞队列
            use_per_session_threads=1,       # 流程的会话设置的线程数。
            allow_soft_placement=True,
            device_count={'CPU': 4, 'GPU': int(use_gpu)})

        '''
        初始化图和config后，开始初始化会话。
        '''
        self.session_tf = tf.Session(graph = self._graph,config = config)
        '''
        预测结果集。
        '''
        predictions = []
        '''
        初始化结束后，开始在计算流图操作,as_default可以覆盖全局
        '''
        with self._graph.as_default():
            tf.set_random_seed(self.seed)
            random_state = np.random.RandomState(self.seed)
            self._xs = SparseMatrix()                                              # 初始化稀疏矩阵
            self._lr = tf.placeholder(tf.float32,shape = [])                       # 初始化学习率
            self.build_model(n_features)                                           # 初始化模型参数
            self.trainADam = tf.train.AdamOptimizer(self._lr).minimize(self.loss)  # 优化器
            self.session_tf.run(tf.global_variables_initializer())                 # 初始化主要参数后，开始run，run里是我们的初始化参数即可
            bs = self.batch_size                                                   # 跑之前，设置batch
            lr = self.learning_rate                                                # 学习率
            for n_epoch in range(self.n_epoch):
                '''
                如果有冗余或者n_epoch为0时，就提交日志信息。开始verbos都是True,也就是默认开始是打印log,如果你不想打印，改成False
                '''
                if verbose and n_epoch == 0:
                    logging.info(memory_info())
                t0 = time.time()
                n = len(y)                                                         # 预测值数量
                indices = random_state.permutation(n)                              # 初始化随机数
                '''
                 初始化就是每次样本的索引
                 以batch为间隔，拆分我们的预测数，然后每次随机化，这样，可以按照
                 batch数间隔随机化，本质就是拆数据
                '''
                index_batches = [indices[idx:idx+bs] for idx in range(0,n,bs)]
                '''
                将索引赋值到我们的样本和目标值上
                '''
                batches = ((X[batch_indices,:],y[batch_indices])for batch_indices in index_batches)
                train_loss = 0                                                     # 初始化训练集损失
                for _x,_y in batches:
                    feed_dict = self._x_feed_dict(_x)
                    feed_dict[self.predict] = _y
                    feed_dict[self._lr] = lr
                    '''
                    初始化训练集的损失值，记住第一个_是预测值，是我们要求的值
                    '''
                    _,loss = self.session_tf.run(
                        [self.trainADam,self.loss],feed_dict = feed_dict
                    )
                    train_loss += loss / len(index_batches)
                print_time = True
                print(X_valid)
                if X_valid is not None:
                    assert not to_predict
                    if y_valid is not None:
                        feed_dict = self._x_feed_dict                             # 二值化
                        feed_dict[self.predict] = y_valid_scaled                  # 预测值进行标准化
                        '''
                        初始化验证集的预测值和损失值
                        '''
                        valid_predict,valid_loss = self.session_tf.run(
                            [self.output,self.loss],feed_dict = feed_dict
                        )
                        valid_predict = self._invert_target(valid_predict)
                        valid_rmsle = get_rmsle(y_valid,valid_predict)
                        dt = time.time() - t0
                        print_time = False
                        print('第{}次的train_loss为: {:.5f}, valid_loss为: {:.5f}, valid_rmsle为: {:.5f}, 时间为: {:.1f}'.format(n_epoch,train_loss,valid_loss,valid_rmsle,dt))
                    else:
                        valid_predict = self.predict(X_valid)
                    predictions.append(valid_predict)
                elif to_predict:
                    predictions.append([self.predict(x) for x in to_predict])
                if print_time:
                    dt = time.time() - t0
                    print('第{}次train_loss为: {:.5f}, 时间为: {:.1f} s'.format(n_epoch,train_loss,dt))
                if n_epoch < self.decay_epochs:
                    bs *= self.bs_decay
                    lr *= self.lr_decay
                if verbose:
                    logging.info(memory_info())
        return self

    '''
    预测
    '''
    def predict(self, X, batch_size=2 ** 13):
        assert self.is_fitted, "模型没有拟合，无法预测"

        ys = []
        for idx in range(0, X.shape[0], batch_size):
            ys.extend(self.session_tf.run(
                self.output, self._x_feed_dict(X[idx: idx + batch_size])))
        return self._invert_target(ys)  #最后别忘了返回标准化预测值

    '''
    预测隐层,预测后用 np.concatenate 沿现有轴加入一系列数组。
    这个是我们的特殊函数，我们可以预测每个hidden的特征结果。
    '''
    def predict_hidden(self, X, batch_size=2 ** 13):
        hidden = []
        for idx in range(0, X.shape[0], batch_size):
            hidden.append(self.session_tf.run(
                self._hidden, self._x_feed_dict(X[idx: idx + batch_size])))
        return np.concatenate(hidden)

'''
下面才是我们真正调用的神经网络
'''

'''
将损失值改为 huber_loss 去创建模型
Huber loss是为了增强平方误差损失函数（squared loss function）
对噪声（或叫离群点，outliers）的鲁棒性（安全稳定性）提出的。
如果想更深了解，可以去查资料。
'''
class RegressionHuber(Regression):
    def build_model(self, n_features: int):
        super(RegressionHuber,self).build_model(n_features)
        self._loss = tf.losses.huber_loss(self.output, self.predict,
                                          weights=2.0, delta=1.0)

'''
将损失值乘以我们的b除以特征数的稀疏，用了最小二乘法。本质也是修改了损失函数。
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.predict)) 这个方法在未来tensor可能被废弃。大概思路也是这样，不停地求softmax损失函数，做交叉熵。
最后取平均值即可。
'''
class RegressionClf(Regression):
    def build_model(self, n_features: int):
        self.predict = tf.placeholder(tf.float32, shape=[None, self.n_bins])
        self.build_hidden(n_features)
        logits = linear(self._hidden, [self.n_hidden[-1], self.n_bins], 'l_last')
        self._output = tf.nn.softmax(logits)
        loss_scale = 6 / self.n_bins
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.predict)) * loss_scale
        self.add_regularization()

    def _scale_target(self, ys):
        return binarize(np.array(ys), self._get_percentiles(ys))

    def _get_percentiles(self, ys):
        if not hasattr(self, 'percentiles'):
            self.percentiles = get_percentiles(ys, self.n_bins)
        return self.percentiles

    def _invert_target(self, ys):
        mean_percentiles = get_mean_percentiles(self.percentiles)
        return (mean_percentiles * ys).sum(axis=1)





