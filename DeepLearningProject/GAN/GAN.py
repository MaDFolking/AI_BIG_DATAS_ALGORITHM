Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @MaDFolking
Sign out
0
0 0 MaDFolking/AI_BIG_DATAS_ALGORITHM
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights  Settings
AI_BIG_DATAS_ALGORITHM/DeepLearningCode/GAN/对抗生成网络案例代码
8287493  a day ago
@MaDFolking MaDFolking Create 对抗生成网络案例代码
     
274 lines (235 sloc)  12.5 KB
Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @MaDFolking
Sign out
0
0 0 MaDFolking/AI_BIG_DATAS_ALGORITHM
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights  Settings
AI_BIG_DATAS_ALGORITHM/GAN_MODEL_TEST
28416c5  on 14 Jun
@MaDFolking MaDFolking Create GAN_MODEL_TEST
     
256 lines (220 sloc)  12.1 KB
import numpy as np
import argparse
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation  #
import seaborn as sns  #可视化库

#案例要求:将一个噪音数据生成的曲线无限接近我们的目标曲线。
#流程: (1)D：判别网络: 对抗生成的网络G(X)和真实输入X分别用判别网络D进行判别。我们希望D能判别出G(X)是假的，直接输出0，对于真实数据X输出1
#(2)G:对抗生成网络，生成无限接近真实值X，最终让D判别不出来。
#(3)为了增强D网络，要预先训练参数，先判断什么是真实数据，什么不是真实。所以要先生成一个网络D_ploy让其区分真实与否的数据。

#流程总结:
#(1)先创建好G网络和D网络的shape,将网络每个shape各个值初始化好。
#(2)创建好G网络和D网络的loss,G网络需要骗过D网络，D网络需要识别出G网络和真实值差距。
#利用公式即可
#(3)训练train,传入俩组数据，一个是真实数据集，一个是噪音的随机初始化数据。将噪音数据调入到梯度下降优化函数来进行求解
#(4)不断迭代优化最终形成效果。

#构造模型
class GAN:
    def __init__(self,data,gen,num_steps,batch_size,log_every):
        self.data = data
        self.gen  =gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4  #隐层神经元层数
        self.learning_rate = 0.03
        self.createModel()  #设置好参数后直接调用创建模型函数

    def createModel(self):
        #先构造D_pre网络用来判别哪个属于真模型，哪个属于假模型
        #利用placeholder创建,这个地方主要是拿到初始化参数
        with tf.variable_scope('D_pre'):
            #输入是个一维点,所以第二个维度设置为1
            self.pre_input  = tf.placeholder(tf.float32,shape=(self.batch_size,1))
            #label是真实值，用来跟预测值D_pre比较的
            self.pre_labels  = tf.placeholder(tf.float32,shape=(self.batch_size,1))
            #将输入数据和神经元进行初始化
            D_pre = discriminator(self.pre_input,self.mlp_hidden_size)
            #定义loss,我们需要计算预测值跟真实值的差异,再开根取平均值就是loss值。
            self.pre_loss = tf.reduce_mean(tf.square(D_pre-self.pre_labels))
            #根据loss来定义优化器求解
            self.pre_opt = optimizer(self.pre_loss,None,self.learning_rate)

        #真正的G网络--对抗生成网络,G网络是通过一个噪音的输入，生成一个真实的数据分布。
        with tf.variable_scope("Gen"):
            self.z = tf.placeholder(tf.float32,shape=(self.batch_size,1))
            self.G = generator(self.z,self.mlp_hidden_size)  #最终输出结果

        #真正的D网络--判别网络,有俩个输入:真实数据和生成网络数据
        with tf.variable_scope("Disc") as scope:
            self.x = tf.placeholder(tf.float32,shape=(self.batch_size,1))
            self.D1 = discriminator(self.x,self.mlp_hidden_size) #真实数据
            scope.reuse_variables()  #重用变量，也就是上面的G,z之类的在这里接着使用
            self.D2 = discriminator(self.G,self.mlp_hidden_size) #对抗网络数据

        #D网络和G网络的损失函数
        self.loss_d = tf.reduce_mean(-tf.log(self.D1)-tf.log(1-self.D2)) #判别真实输入，希望损失值越低越好self.D1是真实数，希望趋近0，self.D2是生成，希望趋近1，这个要好好思考。
        self.loss_g = tf.reduce_mean(-tf.log(self.D2)) #目的是骗过D网络,希望self.D2趋近于1.

        #下面是获取初始化参数,预测，D,G
        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
        #定义优化求解器。最后用梯度下降不断的优化。这个是学习率不断衰减。
        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    #训练网络
    def train(self):
        #先定义session域，初始化tf进行run
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            # pretraining discriminator
            #先训D_pre网络训练出好的w和b,将其拥有分别真实与生成网络区别，再拿这个参数去训练真正的D网络
            #先随机数据生成和label，站住shape的坑。d和label是占据shape的值，label是根据d生成的高斯分布值
            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                #session的run进行迭代
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            #获取训练好的参数
            self.weightsD = session.run(self.d_pre_params)
            # copy weights from pre-training over to new D network
            #将参数拿来进行初始化。本质是将参数深拷贝，也就是生成D网络
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))
            #训练G网络
            for step in range(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size)  #真实数据
                z = self.gen.sample(self.batch_size)   #噪音数据,通过G来生成数据点。
                #下面是生成格式
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                #上面做完后需要更新z数据集
                z = self.gen.sample(self.batch_size)
                #迭代优化操作
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })
                #打印相应的loss值
                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
                if step % 100 == 0 or step == 0 or step == self.num_steps - 1:
                    self._plot_distributions(session)

    def _samples(self, session, num_points=10000, num_bins=100):
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return pd,pg

    def _plot_distributions(self, session):
        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


#G网络输出函数，这里用softplus表示，只用2层毕竟是测试，而且只是1维点。
def generator(input,hDim):
    # 计算softplus：log(exp(features) + 1) f(x) = ln(1+e^x)
    #softplus图像含义就是生成一个过(0,1)点的一直大于0的Log函数
    h0 = tf.nn.softplus(linear(input,hDim,'g0'))
    h1 = linear(h0,1,'g1')
    return h1


#将输入数据和神经元层数,根据深度学习规律，第二层的输入=第一层的输出
def discriminator(input,hDim):
    #里面第二个就是神经元个数，我们设置为隐层数的2倍。用linear函数控制初始化
    h0 = tf.tanh(linear(input,hDim*2,'d0'))
    h1 = tf.tanh(linear(h0,hDim*2,'d1'))
    h2 = tf.tanh(linear(h1,hDim*2,'d2'))
    h3 = tf.sigmoid(linear(h2,hDim*2,'d3'))  #最后一层用sigmoid分类。当做输出层
    return h3

#控制初始化函数
def linear(input,outPutDim,scope=None,stddev=1.0):
    #定义随机初始化,这是w参数,一般w都是这种随机高斯初始化。
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)  #b参数初始化为0的写法
    with tf.variable_scope(scope or 'linear'):  #变量域
        # 第一个是变量名，第二个是shape,w是输入的列*输出神经元的矩阵。我们这里输入列是1，所以取第二个shape
        # initializer是初始化方式。b就是输出的神经元个数，因为b不属于矩阵了。与神经元一一对应。
        w = tf.get_variable('w',[input.get_shape()[1],outPutDim],initializer=norm)
        b = tf.get_variable('b',[outPutDim],initializer=const)
        return tf.matmul(input,w)+b  #这是矩阵乘法函数

#优化器求解函数，根据loss,我们的策略是学习率不断衰减。
def optimizer(loss,varList,initialLearningRate):
    decay = 0.95  #衰减策略
    num_decay_steps = 150  #说明网络每迭代150次，学习率就衰减一次
    batch = tf.Variable(0)
    #这个函数是学习率衰减的函数。初始学习率,batch初始值，没多少次衰减，衰减策略，
    learning_rate = tf.train.exponential_decay(
        initialLearningRate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    #创建求解器,下面是梯度下降的求解器,然后调用梯度下降的最小化函数，里面参数是loss损失函数
    #batch,
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=varList
    )
    return optimizer






#大G网络初始的数据图与噪音点。
class GeneratorDistribution:
    def __init__(self,range):
        self.range = range
    def sample(self,N):
        return np.linspace(-self.range,self.range,N)+np.random.random(N)*0.01


#真实数据集
class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self,N):
        samples = np.random.normal(self.mu,self.sigma,N)  #第一个参数是均值，也就是中心点，第二个0~1表示宽度，第三个是个数。返回array或标量。
        samples.sort()
        return samples

#主函数
def main(args):
    model = GAN(
        DataDistribution(),  #真实数据，里面遵循高斯分布。 #normal第一个参数是均值也就是正态分布的中心点最高点，第二个是0~1表示宽度，第三个是个数，返回array或者标量。
        GeneratorDistribution(range=8), #大G网络初始的数据图与噪音点。
        args.num_steps,  #迭代次数
        args.batch_size,  #batch数量也就是样本数
        args.log_every    #隔多少次打印一次loss
    )
    model.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About
Press h to open a hovercard with more details.
