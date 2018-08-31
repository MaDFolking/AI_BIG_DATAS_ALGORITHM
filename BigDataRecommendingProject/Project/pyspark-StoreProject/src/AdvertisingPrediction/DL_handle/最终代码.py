

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, train_test_split,ShuffleSplit
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection._split import check_cv
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew, kurtosis
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import LinearRegression
import matplotlib
import itertools
import operator
import warnings


from data.data_test.DCN_tf_model import DCNHuber,DCNClf,DCNLos,DCNMse,DCN
from functools import partial

from matplotlib import pylab as plt
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PATH1 = ''
PATH2 = 'D:\\kaggle比赛\\装潢公司广告点击率项目模块\\data\\data_train\\'
PATH3 = PATH2+'data_train_tmp\\'
PATH_TARGET = PATH2+'data_train_target\\'
PATH_TRAIN_USER_FEATURE = PATH2+'data_train_user_feature\\'
PATH_TRAIN_USER_OF_CTR_FEATURE = PATH2+'data_train_user_of_ctr_feature\\'
PATH_TRAIN_CTR_FEATURE =PATH2+'data_train_CTR_feature\\'
PATH_TRAIN_CTR_OF_TIME_FEATURE = PATH2+'data_train_ctr_of_time_feature\\'
KF_SIZE = 10

'''

二值化:https://blog.csdn.net/stdcoutzyx/article/details/50926174

这个样的公式让我想起跟一个大神聊天时谈到的问题，比如，在我之前Google点击率预估那篇博文中提到的一种网络压缩方法，
即不适用32bit的浮点数而是使用16bit格式的数字。既然有压缩，那么就会遇到精度问题，比如如果压缩后的数值表示精度能到0.01，
而更新的梯度的值没到这个精度，比如0.001，此时该如何更新这个值？
答案就是用一定的概率去更新这个值
'''


import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class DCN(BaseEstimator, TransformerMixin):

    def __init__(self, cate_feature_size, field_size,numeric_feature_size,
                 embedding_size=8,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True,cross_layer_num=3):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.cate_feature_size = cate_feature_size
        self.numeric_feature_size = numeric_feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size
        self.deep_layers = deep_layers
        self.cross_layer_num = cross_layer_num
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result,self.valid_result = [],[]

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None,None],
                                             name='feat_index')
            self.feat_value = tf.placeholder(tf.float32,
                                           shape=[None,None],
                                           name='feat_value')

            self.numeric_value = tf.placeholder(tf.float32,[None,None],name='num_value')

            self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool,name='train_phase')

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            self.embeddings = tf.multiply(self.embeddings,feat_value)

            self.x0 = tf.concat([self.numeric_value,
                                 tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])]
                                ,axis=1)


            # deep part


            self.y_deep = tf.nn.dropout(self.x0,self.dropout_keep_deep[0])

            for i in range(0,len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])


            # cross_part
            self._x0 = tf.reshape(self.x0, (-1, self.total_size, 1))
            x_l = self._x0
            for l in range(self.cross_layer_num):
                x_l = tf.tensordot(tf.matmul(self._x0, x_l, transpose_b=True),
                                    self.weights["cross_layer_%d" % l],1) + self.weights["cross_bias_%d" % l] + x_l

            self.cross_network_out = tf.reshape(x_l, (-1, self.total_size))


            # concat_part
            concat_input = tf.concat([self.cross_network_out, self.y_deep], axis=1)

            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["deep_layer_%d" % i])
                for i in range(self.cross_layer_num):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["cross_layer_%d" % i])


            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)


            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)





    def _initialize_weights(self):
        weights = dict()

        #embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01),
            name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.cate_feature_size,1],0.0,1.0),name='feature_bias')


        #deep layers
        num_layer = len(self.deep_layers)
        glorot = np.sqrt(2.0/(self.total_size + self.deep_layers[0]))

        weights['deep_layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(self.total_size,self.deep_layers[0])),dtype=np.float32
        )
        weights['deep_bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32
        )


        for i in range(1,num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["deep_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        for i in range(self.cross_layer_num):

            weights["cross_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.total_size,1)),
                dtype=np.float32)
            weights["cross_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.total_size,1)),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer


        input_size = self.total_size + self.deep_layers[-1]

        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)


        return weights


    def get_batch(self,Xi,Xv,Xv2,y,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end],Xv[start:end],Xv2[start:end],[[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c,d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def predict(self, Xi, Xv,Xv2,y):


        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss


    def fit_on_batch(self,Xi,Xv,Xv2,y):
        feed_dict = {self.feat_index:Xi,
                     self.feat_value:Xv,
                     self.numeric_value:Xv2,
                     self.label:y,
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)

        return loss

    def fit(self, cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train,
            cate_Xi_valid=None, cate_Xv_valid=None, numeric_Xv_valid=None,y_valid=None,
            early_stopping=False, refit=False):

        print(len(cate_Xi_train))
        print(len(cate_Xv_train))
        print(len(numeric_Xv_train))
        print(len(y_train))
        has_valid = cate_Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train, self.batch_size, i)

                self.fit_on_batch(cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch)


            if has_valid:
                y_valid = np.array(y_valid).reshape((-1,1))
                loss = self.predict(cate_Xi_valid, cate_Xv_valid, numeric_Xv_valid, y_valid)
                print("epoch",epoch,"loss",loss)



'''
内存查看
'''
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep = True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep = True)
    usage_mb = usage_b / 1024 ** 2
    return "{:.2f} MB".format(usage_mb)

def load_train_iter():
    # aa = pd.read_csv(PATH1+'train.csv',usecols=['hour'])
    # print(aa.shape[0])
    bb = 40428967
    print(bb//36)


'''
拆分数值型和非数值型函数
'''
def parse_numeric(train_x):
    assert len(train_x)>0
    features = train_x.columns
    numeric_feature = train_x.columns[train_x.dtypes!='object']
    numeric_non_feature = [i for i in features if i not in numeric_feature]
    return features,numeric_feature,numeric_non_feature

'''
非数值型编码
'''
def numeric_non_scale(train_x):
    assert len(train_x)>0
    print("非数值型编码处理开始...")
    print(train_x.dtypes)
    #先LabelEncoder处理
    _,_,numeric_non_feature = parse_numeric(train_x)
    print(numeric_non_feature)
    le = LabelEncoder()
    for i in numeric_non_feature:
        le.fit(list(train_x[i]))
        train_x[i] = le.transform(train_x[i])
    print(train_x.dtypes)
    # one_hot = OneHotEncoder()
    # train_x = one_hot.fit_transform(train_x)
    print(train_x.head())
    print(train_x.shape)
    print(train_x.isnull().sum().sort_values(ascending=False))
    print("非数值型编码处理结束...")
    return train_x

'''
数值类型标准化
'''
def numeric_scale(train_x):
    assert len(train_x)>0
    print("数值标准化处理开始...")
    _,numeric_feature,_ = parse_numeric(train_x)
    numeric_mean = train_x.loc[:,numeric_feature].mean()
    numeric_std = train_x.loc[:,numeric_feature].std()
    train_x.loc[:,numeric_feature] = (train_x.loc[:,numeric_feature]-numeric_mean)/numeric_std
    print(train_x.head())
    print(train_x.shape)
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    print(type(train_x.isnull().sum().sort_values(ascending=False).head()))
    print(type(train_x.isnull().sum().sort_values(ascending=False)[0]))
    #我们这次并没有nan值。所以函数里只是注释一些内容
    if train_x.isnull().sum().sort_values(ascending=False)[0]>0:
        parse_nan(train_x)
    print(train_x.head())
    print(train_x.shape)
    print(train_x.dtypes)
    feature, numeric_feature, _ = parse_numeric(train_x)
    if len(feature)==len(numeric_feature):
        print("数值标准化处理结束...")
        return train_x
    else:
        raise NotImplementedError("数值标准化处理失败...")

'''
去除nan值
'''
def parse_nan(train_x):
    assert len(train_x)>0
    print("异常值处理开始...")
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    # train_x.CompetitionDistance.fillna(train_x.CompetitionDistance.mean(), inplace=True)
    # train_x.Open.fillna(1, inplace=True)
    train_x.fillna(0, inplace=True)
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    print("异常值处理结束...")


'''
特征拼接--针对我们设计的强化学习，去拼接特征。
并存储用户单独的特征数据集，用户与广告类别共同特征拼接后的数据集，广告类别数据集，广告类别在用户不同时间段的数据集。
'''
def feature_union(train_x,i):
    assert len(train_x)>0
    print(train_x.dtypes)
    train_x.astype('float16')
    print(train_x.dtypes)
    '''
    拼接用户与广告类别的特征，这个是用作 state 的状态改变而拼接。
    '''
    train_x['C1_site_id'] = np.add(train_x.C1.values,train_x.site_id.values).astype('float16')
    train_x['C1_app_id'] = np.add(train_x.C1.values, train_x.app_id.values)
    train_x['C1_device_conn_type'] = np.add(train_x.C1.values, train_x.device_conn_type.values)

    train_x['C14_site_id'] = np.add(train_x.C14.values, train_x.site_id.values)
    train_x['C14_app_id'] = np.add(train_x.C14.values, train_x.app_id.values)
    train_x['C14_device_conn_type'] = np.add(train_x.C14.values, train_x.device_conn_type.values)

    train_x['C17_site_id'] = np.add(train_x.C17.values, train_x.site_id.values)
    train_x['C17_app_id'] = np.add(train_x.C17.values, train_x.app_id.values)
    train_x['C17_device_conn_type'] = np.add(train_x.C17.values, train_x.device_conn_type.values)

    train_x['C19_site_id'] = np.add(train_x.C19.values, train_x.site_id.values)
    train_x['C19_app_id'] = np.add(train_x.C19.values, train_x.app_id.values)
    train_x['C19_device_conn_type'] = np.add(train_x.C19.values, train_x.device_conn_type.values)

    train_x['C20_site_id'] = np.add(train_x.C20.values, train_x.site_id.values)
    train_x['C20_app_id'] = np.add(train_x.C20.values, train_x.app_id.values)
    train_x['C20_device_conn_type'] = np.add(train_x.C20.values, train_x.device_conn_type.values)

    train_x['C21_site_id'] = np.add(train_x.C21.values, train_x.site_id.values)
    train_x['C21_app_id'] = np.add(train_x.C21.values, train_x.app_id.values)
    train_x['C21_device_conn_type'] = np.add(train_x.C21.values, train_x.device_conn_type.values)

    '''
    离散时间
    '''
    # train_x['day'] = np.round(train_x.hour % 10000 / 100)
    # train_x['hour1'] = np.round(train_x.hour % 100)
    # train_x['day_hour'] = (train_x.day.values - 21) * 24 + train_x.hour1.values
    # # train_x['day_hour_prev'] = train_x['day_hour'] - 1
    # # train_x['day_hour_next'] = train_x['day_hour'] + 1
    # train_x.drop('day',axis = 1,inplace = True)
    # train_x.drop('hour1', axis = 1,inplace = True)

    '''
    拼接不同时间段的不同类别广告点击率，当做action使用。这步在强化学习时再做。
    '''
    # train_x['C1_hour'] = np.add(train_x.C1.values, train_x.hour.values).astype('float16')
    # train_x['C14_hour'] = np.add(train_x.C14.values, train_x.hour.values)
    # train_x['C17_hour'] = np.add(train_x.C17.values, train_x.hour.values)
    # train_x['C19_hour'] = np.add(train_x.C19.values, train_x.hour.values)
    # train_x['C20_hour'] = np.add(train_x.C20.values, train_x.hour.values)
    # train_x['C21_hour'] = np.add(train_x.C21.values, train_x.hour.values)

    print(train_x.shape)
    train_x.to_csv(PATH3+str(i))
    print("保存成功!")

    # '''
    # 开始保存各个位置特征
    # '''
    # train_x['banner_pos','site_id'].to_csv(PATH_TRAIN_USER_FEATURE + 'train'+str(i)+'.csv',index = False)



from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

def load_data_2(i):
    train = pd.read_csv(PATH3+i)
    return train

def save_data_target():
    data = pd.read_csv('train.csv', iterator=True)
    train = pd.read_csv('train.csv', usecols=['hour'])
    print(train.shape)
    print(train.head())
    print(train.dtypes)
    list_train = []
    train.drop_duplicates(inplace=True)
    for i in train.index:
        list_train.append(i)
    for j in list_train:
        print(j)
    train1 = pd.read_csv('train.csv', usecols=['click'], iterator=True, index_col=False, engine='python')
    while True:
        try:
            for i in list_train:
                chunk = train1.get_chunk(i)
                chunk.to_csv(PATH_TARGET + 'train_target{}.csv'.format(i), index=False, sep=',')
        except StopIteration:
            print("Iteration is stopped.")

'''
加载目标值
'''
def load_data_target(i):
    target = pd.read_csv(PATH_TARGET+i,index_col = False)
    return target

'''
离散化
'''
def numeric_non_dispersed(x_train):
    assert len(x_train) > 0

    list_feature = ['site_category','app_category','device_type','device_conn_type',
                    'C1','C14','C15','C16','C17','C18','C19','C20','C21']
    print(x_train.shape)
    for i in list_feature:
        train_new = pd.get_dummies(i)
        x_train = pd.concat((x_train,train_new),axis = 1)
        print(x_train.shape)
    print(x_train.head())
    x_train.fillna(0,inplace=True)
    print(x_train.isnull().sum().sort_values(ascending=False).head())
    print("离散后数据集为: {}".format(x_train.shape))
    return x_train


def main():
    load_train_iter()

    # files = os.listdir(PATH2)
    # print(files)
    # files = files[6:]
    # print(files)
    # for i in files:
    # X_train = load_data_2('train921985.csv')                              # 加载训练数据
    # # save_data_target()
    # y_train = load_data_target('train_target921985.csv')                  # 加载目标数据
    # # numeric_non_scale(X_train)                                          # 非数值型标准化
    # # numeric_scale(X_train)                                              # 数值型标准化
    # # X_train1 = numeric_non_dispersed(X_train1)                          # 类别型离散化
    # # X_train2 = numeric_non_dispersed(X_train2)
    # #X_train.drop('Unnamed: 0',inplace = True)
    # '''
    # 接下来，要区分连续型和类别型。
    # '''
    # list_total = X_train.columns
    # list_category = ['site_category', 'app_category', 'device_type', 'device_conn_type',
    #                 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    # list_numeric = [i for i in list_total if i not in list_category]
    # print(list_total)
    # print(list_category)
    # print(list_numeric)
    # train_numeric = X_train[list_numeric]
    # train_category = X_train[list_category]
    # print("数值型：",train_numeric.shape)
    # print("类别型：",train_category.shape)
    # len_category = len(list_category)
    # len_numeric = len(list_numeric)
    # dict_category = {}
    # tc = 0
    #
    # '''
    # 下面是求embbed的第二个序号方法。
    # '''
    # train_category_copy = train_category.copy()
    # for i in list_category:
    #     print("每次循环长度：",len(X_train[i]))
    #     us = X_train[i].unique()
    #     print("us: ",us)
    #     dict_category[i] = dict(zip(us, range(tc, len(us) + tc)))
    #     print("dict: ",dict_category[i])
    #     print("tc:",tc)
    #     tc += len(us)
    #     train_category_copy[i] = train_category_copy[i].map(dict_category[i])
    # len_category_size = tc
    # for i in train_category_copy:
    #     print(i)
    # dict_category_index = train_category_copy.values.tolist()
    # for i in dict_category_index:
    #     print("最终索引每一行值为:",i)
    # print("索引长度:",len(dict_category_index))
    # print("索引shape：",np.array(dict_category_index).shape)
    # print(dict_category_index[0])
    # #print(dict_category)
    #
    # batch_size = 1024
    # epoch =30
    # print(type(epoch))
    # kf = KFold(n_splits = KF_SIZE,shuffle = True,random_state = 2018)
    #
    # '''
    # 参数
    # '''
    #
    # dcn_params = {
    #     "embedding_size": 4,
    #     "deep_layers": [32, 32],
    #     "dropout_deep": [0.5, 0.5, 0.5],
    #     "deep_layers_activation": tf.nn.relu,
    #     "epoch": epoch,
    #     "batch_size": batch_size,
    #     "learning_rate": 0.001,
    #     "optimizer_type": "adam",
    #     "batch_norm": 1,
    #     "batch_norm_decay": 0.995,
    #     "l2_reg": 0.1,
    #     "verbose": True,
    #     "random_seed": 2018,
    #     "cross_layer_num": 3
    # }
    #
    # dcn_params["cate_feature_size"] = len_category_size
    # dcn_params["field_size"] = len(dict_category_index[0])
    # dcn_params['numeric_feature_size'] = len_numeric
    # '''
    # 因为DCN中涉及到坐标值，所以在做交叉时，用 enumerate 做出索引值。
    # '''
    # print(kf.split(X_train))
    # print(list(kf.split(X_train,y_train)))
    #
    # for train_idx,valid_idx in kf.split(X_train,y_train):
    #     print(train_idx,valid_idx)
    #     cate_index_train,cate_value_train,numeric_value_train,y_train_ =\
    #         [dict_category_index[i] for i in train_idx],train_category.iloc[train_idx],train_numeric.iloc[train_idx],y_train.iloc[train_idx]
    #     cate_index_valid, cate_value_valid, numeric_value_valid, y_valid_ = \
    #         [dict_category_index[i] for i in valid_idx], train_category.iloc[valid_idx],train_numeric.iloc[valid_idx], y_train.iloc[valid_idx]
    #     dcn_los = DCNLos(**dcn_params)
    #     dcn_mse = DCNMse(**dcn_params)
    #     dcn_hub = DCNHuber(**dcn_params)
    #     dcn_clf = DCNClf(**dcn_params)
    #     y_train_ = y_train_.values
    #     #dcn_los.fit(cate_index_train,cate_value_train,numeric_value_train,y_train_,cate_index_valid, cate_value_valid, numeric_value_valid, y_valid_)
    #     #dcn_mse.fit(cate_index_train, cate_value_train, numeric_value_train, y_train_, cate_index_valid,
    #                 #cate_value_valid, numeric_value_valid, y_valid_)
    #     #dcn_hub.fit(cate_index_train, cate_value_train, numeric_value_train, y_train_, cate_index_valid,
    #                 #cate_value_valid, numeric_value_valid, y_valid_)
    #     dcn_clf.fit(cate_index_train, cate_value_train, numeric_value_train, y_train_, cate_index_valid,
    #                 cate_value_valid, numeric_value_valid, y_valid_)


if __name__ == '__main__':
    main()











