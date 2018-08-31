

import gc
import time
import random
import pickle
import locale
import hashlib
import datetime
import pycountry
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd

from itertools import combinations
from collections import defaultdict
from matplotlib import pylab as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble, linear_model, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv,KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from model_rl_ml import XGBClassifyCV,LGBClassifyCV,LRClassifyCV,SGDClassifyCV,LGB_model,XGB_model


PATH1 = 'D:\\kaggle比赛\\装潢公司广告点击率项目模块\\data\\'
PATH2 = 'D:\\kaggle比赛\\装潢公司广告点击率项目模块\\data\\data_train\\'
PATH3 = PATH2+'data_train_tmp\\'
PATH4 = PATH2+'data_test_tmp\\'
PATH_TARGET = PATH2+'data_train_target\\'
PATH_TRAIN_USER_OF_CTR_FEATURE = PATH2+'data_train_user_of_ctr_feature\\'
PATH_TRAIN_CTR_FEATURE =PATH2+'data_train_CTR_feature\\'
PATH_TRAIN_CTR_OF_TIME_FEATURE = PATH2+'data_train_ctr_of_time_feature\\'
KF_SIZE = 10
SEED = 2018

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

def load_data():
    train = pd.read_csv(PATH1+'train.csv', usecols=['hour'])
    print(train.shape)
    print(train.head())
    print(train.dtypes)
    list_train = []
    train.drop_duplicates(inplace=True)
    print(train.shape)
    print(train.index)
    for i in range(21,31):
        for j in train.index:
            # print(j)
            # print(train.iloc[j])
            j_new = train.loc[j].values
            print(j_new)
            # print(type(j_new))
            # print(j_new.shape)
            s = str(j_new)[5:7]
            # print(s)
            if int(s) > i:
                break
            j_new = int(j_new)
            print(j_new)
            if (14100000+i*100) < j_new:
                train.drop(j,axis = 0,inplace = True)
            else:
                continue

    print(train.shape)
    for i in train.index:
        list_train.append(i)

    for j in list_train:
        print(j)
    train1 = pd.read_csv(PATH1+'train.csv',iterator = True,index_col=False,engine = 'python')
    while True:
        try:
            for i in list_train:
                chunk = train1.get_chunk(i)
                chunk.to_csv(PATH2+'train{}.csv'.format(i), index=False, sep=',')
        except StopIteration:
            print("Iteration is stopped.")

'''
加载训练集
'''
def load_train_data():
    train_x = pd.read_csv(PATH2+'train4122995.csv')
    print(train_x.shape)
    print(train_x.columns)
    print(train_x[['C1','C20']])
    train_y = train_x.click
    print(train_y)
    train_x.drop('click',axis = 1,inplace = True)
    print(train_x.shape," ",train_y.shape)
    return train_x,train_y

'''
拆分数值型和非数值型函数
'''
def parse_numeric(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
    features = train_x.columns
    numeric_feature = train_x.columns[train_x.dtypes!='object']
    numeric_non_feature = [i for i in features if i not in numeric_feature]
    return features,numeric_feature,numeric_non_feature

'''
由于内存原因，我就不One-hot+PCA+随机森林分类了，具体可以看我的kaggle比赛项目和销量预测项目如何使用这个步骤。
'''

'''
非数值型编码
'''
def numeric_non_scale(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
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
    if train_x is None:
        raise ValueError("输入值不能为空!")
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
    if train_x is None:
        raise ValueError("输入值不能为空!")
    print("异常值处理开始...")
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    # train_x.CompetitionDistance.fillna(train_x.CompetitionDistance.mean(), inplace=True)
    # train_x.Open.fillna(1, inplace=True)
    train_x.fillna(0, inplace=True)
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    print("异常值处理结束...")

'''
针对XGB,LGB选取的树形特征,我们在机器学习篇章里已经做完特征工程，所以这里我就直接用树形和线性开始构造他们各自的强特征了。
'''
from xgboost import XGBRegressor
from xgboost import XGBClassifier
'''
下面是构造适合XGB,LGB的树形特征，记住，特征可以自己选择，因为我们是强化学习，要保持一个动态，这也是我们
要更新的参数。
'''
class createTreeFeatures:
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd = random_state
        self.n = n_neighbours
        self.max_elts = max_elts
        self.verbose = verbose
        self.neighbours = []
        self.clfs = []

    def fit(self, train, y):
        if self.rnd != None:
            random.seed(self.rnd)
        if self.max_elts == None:
            self.max_elts = len(train.columns)
        list_vars = train.columns

        lastscores = np.zeros(self.n) + 1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars = list_vars[self.n:]

        for elt in list_vars:
            indice = 0
            scores = []
            for elt2 in self.neighbours:
                if len(elt2) < self.max_elts:
                    clf = XGBClassifier(n_estimators=1000,n_jobs=-1,nthread=-1)
                    clf.fit(train[elt2 + [elt]], y,
                    eval_set=[(train[elt2 + [elt]], y)],
                    eval_metric='auc',
                    early_stopping_rounds=15,
                    verbose=True)
                    scores.append(metrics.mean_squared_error(y, clf.predict(train[elt2 + [elt]])))
                    indice = indice + 1
                else:
                    scores.append(lastscores[indice])
                    indice = indice + 1
            gains = lastscores - scores
            if gains.max() > 0:
                temp = gains.argmax()
                lastscores[temp] = scores[temp]
                self.neighbours[temp].append(elt)

        indice = 0
        for elt in self.neighbours:
            clf = XGBClassifier(n_estimators=50, n_jobs=-1, nthread=-1)
            clf.fit(train[elt2 + [elt]], y,
                    eval_set=[(train[elt2 + [elt]], y)],
                    eval_metric='auc',
                    early_stopping_rounds=15)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice = indice + 1

    def transform(self, train):
        indice = 0
        for elt in self.neighbours:
            train[str(elt[0]) + '_' + str(elt[1])] = self.clfs[indice].predict(train[elt])
            indice = indice + 1
        print(train.shape)
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)


'''
下面是构造适合LR,SGD 的 线性特征，记住，特征可以自己选择，因为我们是强化学习，要保持一个动态，这也是我们
要更新的参数。
'''
from sklearn.linear_model import Lasso
class createLinearFeatures:
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd = random_state
        self.n = n_neighbours
        self.max_elts = max_elts
        self.verbose = verbose
        self.neighbours = []
        self.clfs = []

    def fit(self, train, y):
        if self.rnd != None:
            random.seed(self.rnd)
        if self.max_elts == None:
            self.max_elts = len(train.columns)
        list_vars = train.columns

        lastscores = np.zeros(self.n) + 1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars = list_vars[self.n:]

        for elt in list_vars:
            indice = 0
            scores = []
            for elt2 in self.neighbours:
                if len(elt2) < self.max_elts:
                    clf = LogisticRegression(fit_intercept=False, n_jobs=-1)
                    clf.fit(train[elt2 + [elt]], y)
                    scores.append(metrics.mean_squared_error(y, clf.predict(train[elt2 + [elt]])))
                    indice = indice + 1
                else:
                    scores.append(lastscores[indice])
                    indice = indice + 1
            gains = lastscores - scores
            if gains.max() > 0:
                temp = gains.argmax()
                lastscores[temp] = scores[temp]
                self.neighbours[temp].append(elt)

        indice = 0
        for elt in self.neighbours:
            clf = LogisticRegression(fit_intercept=False, n_jobs=-1)
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice = indice + 1

    def transform(self, train):
        indice = 0
        for elt in self.neighbours:
            train[str(elt[0]) + '_' + str(elt[1])] = self.clfs[indice].predict(train[elt])
            indice = indice + 1
        print(train.shape)
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)

'''
奖励值
'''
def get_reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    R = np.sign(R2) * np.sqrt(abs(R2))
    return (R)



def main():
    #load_data()                                        # 拆分数据集
    train_x, train_y = load_train_data()                # 加载训练集
    # train_x = numeric_non_scale(train_x)                # 类别型数值化
    # train_x = numeric_scale(train_x)                    # 数值型标准化
    #
    # cols = [i for i in train_x.columns if i not in ['id']]
    # train_x = train_x[cols]
    # print("adding linear new features")
    # feature_linear = createLinearFeatures(n_neighbours=5, max_elts=2, verbose=True, random_state=SEED)
    # feature_linear.fit(train_x, train_y)
    # train_linear = feature_linear.transform(train_x)
    # linear_columns = train_linear.columns
    # train_linear.to_csv(PATH_TARGET+'train_linear_001.csv',index = False)
    '''
    加载线性数据。
    '''
    train_linear = pd.read_csv(PATH_TARGET+'train_linear_001.csv')
    train_linear = pd.concat((train_x['id'],train_linear),axis = 1)
    linear_columns = train_linear.columns
    print(linear_columns)

    linear_columns_new = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                          'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                          'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
                          'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'hour_app_id',
                          'C1_site_category', 'banner_pos_app_domain', 'site_id_device_id',
                          'site_domain_app_category']

    # print("adding tree new features")
    # feature_tree = createTreeFeatures(n_neighbours=5, max_elts=2, verbose=True, random_state=SEED)
    # feature_tree.fit(train_x, train_y)
    # train_tree = feature_tree.transform(train_x)
    # tree_columns = train_tree.columns
    # print(tree_columns)

    '''
    总模型集合，奖励池，特征集合
    '''
    total_models = []
    total_columns = []
    total_rewards = []

    '''
    LR模型集合，奖励池，特征集合
    '''
    lr_models = []
    lr_columns = []
    lr_rewards = []

    '''
    我们先用LR,SGD来做实验，因为线性的比较快。
    '''
    '''
    combinations 可以按照顺序，进行俩俩拼接组合。
    '''
    # lr_params = {
    #     'solver':'sag',
    #     'class_weight':'balanced',
    #     'random_state':SEED,
    #     'n_jobs':-1
    # print(train_linear.shape," ",train_y.shape)
    # for (col1,col2) in combinations(linear_columns_new,2):
    #     print(col1," ",col2)
    #     lr_model = LogisticRegression(solver = 'sag',class_weight='balanced',random_state=SEED,n_jobs=-1)
    #     lr_model.fit(train_linear[[col1,col2]],train_y)
    #     lr_models.append(lr_model)
    #     lr_columns.append([col1,col2])
    #
    #     '''
    #     预测值列表
    #     '''
    #     y_pre = pd.Series(lr_model.predict(train_linear[[col1,col2]]),index = train_linear.index,name = 'lr_pre')
    #     tmp_train = pd.concat([train_linear,y_pre],axis = 1)
    #     reward = tmp_train.hour.map(tmp_train.groupby('hour').apply(lambda x:get_reward(train_y,x['lr_pre'])))
    #     reward.name = 'reward'
    #     '''
    #     shift: https://blog.csdn.net/kizgel/article/details/78333833
    #     '''
    #     tmp_train = pd.concat([tmp_train,reward],axis = 1)
    #     reward_shift = tmp_train.groupby('id').apply(lambda x:x['reward'].shift(1)).fillna(0)
    #     lr_rewards.append(reward_shift)
    #     del y_pre,tmp_train,reward
    #     gc.collect()
    #
    # '''
    # 取前7个好的模型
    # '''
    # lr_num = 7
    # target_select = np.array(lr_rewards).T
    # target_select = np.argmax(target_select , axis = 1)
    #
    # print("selecting best models:")
    # print(pd.Series(target_select).value_counts().head(lr_num))
    # lr_index = pd.Series(target_select).value_counts().head(lr_num).index
    # for elt in lr_index:
    #     total_models.append(lr_models[elt])
    #     total_columns.append(lr_columns[elt])
    #     total_rewards.append(lr_rewards[elt])
    #
    # del lr_models
    # del lr_columns
    # del lr_rewards
    # gc.collect()


    '''
    SGD模型集合,奖励池,特征集合
    '''
    # sgd_models = []
    # sgd_columns = []
    # sgd_rewards = []
    #
    # sgd_params = {
    #     'loss': "log",
    #     'penalty': "l2"
    # }
    # nums = 0
    #
    # for (col1,col2) in combinations(linear_columns_new,2):
    #     sgd_model = SGDClassifier(loss = 'log',penalty = 'l2')
    #     sgd_model.fit(train_linear[[col1,col2]],train_y)
    #     sgd_models.append(sgd_model)
    #     sgd_columns.append([col1,col2])
    #
    #     '''
    #     预测值列表
    #     '''
    #     y_pre = pd.Series(sgd_model.predict(train_linear[[col1,col2]]),index = train_linear.index,name = 'sgd_pre')
    #     tmp_train = pd.concat([train_linear,y_pre],axis = 1)
    #     reward = tmp_train.hour.map(tmp_train.groupby('hour').apply(lambda x:get_reward(train_y,x['sgd_pre'])))
    #     reward.name = 'reward'
    #     '''
    #     shift: https://blog.csdn.net/kizgel/article/details/78333833
    #     '''
    #     tmp_train = pd.concat([tmp_train,reward],axis = 1)
    #     reward_shift = tmp_train.groupby('id').apply(lambda x:x['reward'].shift(1)).fillna(0)
    #     sgd_rewards.append(reward_shift)
    #     del y_pre,tmp_train,reward
    #     gc.collect()
    #
    # '''
    # 取前7个好的模型
    # '''
    # sgd_num = 7
    # target_select_sgd = np.array(lr_rewards).T
    # target_select_sgd = np.argmax(target_select_sgd , axis = 1)
    #
    # print("selecting best models:")
    # print(pd.Series(target_select_sgd).value_counts().head(sgd_num))
    # sgd_index = pd.Series(target_select_sgd).value_counts().head(sgd_num).index
    # for elt in sgd_index:
    #     total_models.append(sgd_models[elt])
    #     total_columns.append(sgd_columns[elt])
    #     total_rewards.append(sgd_rewards[elt])
    #
    # del sgd_models
    # del sgd_columns
    # del sgd_rewards
    # gc.collect()




    '''
    XGB模型集合,奖励池,特征集合
    '''
    # xgb_models = []
    # xgb_columns = []
    # xgb_rewards = []
    #
    # print("XGB训练开始...")
    # max_leaf_nodes = [100]
    # min_child_weight = [1, 3, 5, 10]
    # # for i in max_leaf_nodes:
    # #     for j in min_child_weight:
    # print("fitting XGB tree model with ", max_leaf_nodes, min_child_weight[0])
    # model = XGB_model(max_leaf_nodes=10, min_child_weight=3)
    # model.fit(train_linear, train_y)
    # print("XGB feature importance")
    # print(pd.DataFrame(model.feature_importance(), index=linear_columns).sort_values(by=[0]).tail(20))
    # print(" ")
    # xgb_models.append(model)
    # xgb_columns.append(linear_columns_new)
    # y_pred = pd.Series(model.predict(train_linear), index=train_linear.index, name="y_pre_xgb")
    # tmp_train = pd.concat((train_linear[['id', 'hour']], train_y), axis=1)
    # tmp_train = pd.concat((tmp_train, y_pred), axis=1)
    # reward = tmp_train.hour.map(
    #     tmp_train.groupby('hour').apply(lambda x: get_reward(train_y, x['y_pre_xgb'])))
    # print("奖励值为:", reward)
    # reward.name = "reward"
    # tmp_train = pd.concat([tmp_train, reward], axis=1)
    # print(tmp_train.shape)
    # reward_shift = tmp_train.groupby('id').apply(lambda x: x['reward'].shift(1)).fillna(0)
    # print("reward_shift:", reward_shift)
    # xgb_rewards.append(reward_shift)
    # del y_pred, tmp_train, reward
    # gc.collect()
    #
    # XG_to_keep = 3
    # target_xgb_selector = np.array(xgb_rewards).T
    # target_xgb_selector = np.argmax(target_xgb_selector, axis=1)
    # print("selecting best models:")
    # print(pd.Series(target_xgb_selector).value_counts().head(XG_to_keep))
    # to_xgb_keep = pd.Series(target_xgb_selector).value_counts().head(XG_to_keep).index
    # for elt in to_xgb_keep:
    #     total_models.append(xgb_models[elt])
    #     total_columns.append(xgb_columns[elt])
    #     total_rewards.append(xgb_rewards[elt])
    # del xgb_models
    # del xgb_columns
    # del xgb_rewards
    # gc.collect()
    #
    # '''
    # LGB模型集合,奖励池，特征集合
    # '''
    # lgb_models = []
    # lgb_columns = []
    # lgb_rewards = []
    #
    # num_leaves = [70]
    # feature_fractions = [0.2, 0.6, 0.8]
    # bagging_fractions = [0.7]
    #
    # # for num_leaf in num_leaves:
    # #   for feature_fraction in feature_fractions:
    # #      for bagging_fraction in bagging_fractions:
    # # print("fitting LGB tree model with ", num_leaf, feature_fraction, bagging_fraction)
    # model = LGB_model(num_leaves=70, feature_fraction=0.8,
    #                   bagging_fraction=0.7)
    # model.fit(train_linear, train_y)
    # print("LGB feature importance")
    # print(pd.DataFrame(model.feature_importance(), index=linear_columns).sort_values(by=[0]).tail(20))
    # print(" ")
    # lgb_models.append(model)
    # lgb_columns.append(linear_columns_new)
    # y_pred = pd.Series(model.predict(train_linear), index=train_linear.index, name="y_pre_lgb")
    # tmp_train = pd.concat((train_linear[['id', 'hour']], train_y), axis=1)
    # tmp_train = pd.concat((tmp_train, y_pred), axis=1)
    # reward = tmp_train.hour.map(
    #     tmp_train.groupby('hour').apply(lambda x: get_reward(train_y, x['y_pre_lgb'])))
    # print("奖励值为:", reward)
    # reward.name = "reward"
    # tmp_train = pd.concat([tmp_train, reward], axis=1)
    # print(tmp_train.shape)
    # reward_shift = tmp_train.groupby('id').apply(lambda x: x['reward'].shift(1)).fillna(0)
    # print("reward_shift:", reward_shift)
    # lgb_rewards.append(reward_shift)
    # del y_pred, tmp_train, reward
    # gc.collect()
    #
    # LG_to_keep = 3
    # target_lgb_selector = np.array(lgb_rewards).T
    # target_lgb_selector = np.argmax(target_lgb_selector, axis=1)
    # print("selecting best models:")
    # print(pd.Series(target_lgb_selector).value_counts().head(LG_to_keep))
    # to_lgb_keep = pd.Series(target_lgb_selector).value_counts().head(LG_to_keep).index
    #
    # for elt in to_lgb_keep:
    #     total_models.append(lgb_models[elt])
    #     total_columns.append(lgb_columns[elt])
    #     total_rewards.append(lgb_rewards[elt])
    #
    # del lgb_models
    # del lgb_columns
    # del lgb_rewards
    # gc.collect()

    print("LGB训练结束......")


    '''
    随机森林
    '''
    rf_models = []
    rf_columns = []
    rf_rewards = []

    rf_model = ensemble.RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=SEED, verbose=2)
    rf_model.fit(train_linear, train_y)
    print(pd.DataFrame(rf_model.feature_importances_, index=linear_columns).sort_values(by=[0]).tail(20))
    for elt in rf_model.estimators_:
        rf_models.append(elt)
        rf_columns.append(linear_columns)

        y_pred = pd.Series(elt.predict(train_linear), index = train_linear.index, name="y_pre_rf")
        print(y_pred.shape)
        print(y_pred.head())
        tmp_train = pd.concat((train_linear[['id','hour']],train_y),axis = 1)
        tmp_train = pd.concat((tmp_train, y_pred), axis = 1)
        reward = tmp_train.hour.map(
            tmp_train.groupby('hour').apply(lambda x: get_reward(train_y, x['y_pre_rf'])))
        reward.name = "reward"
        tmp_train = pd.concat([tmp_train, reward], axis=1)
        reward_shift = tmp_train.groupby('id').apply(lambda x: x['reward'].shift(1)).fillna(0)
        print(reward_shift)
        rf_rewards.append(reward_shift)
        del y_pred, tmp_train, reward
        gc.collect()

    rf_to_keep = 3
    target_rf_selector = np.array(rf_rewards).T
    target_rf_selector = np.argmax(target_rf_selector, axis=1)

    print("selecting best models:")
    print(pd.Series(target_rf_selector).value_counts().head(rf_to_keep))
    to_rf_keep = pd.Series(target_rf_selector).value_counts().head(rf_to_keep).index
    to_rf_keep = pd.Series(target_rf_selector).index
    aaa = 0
    for elt in to_rf_keep:
        total_models.append(rf_model[0])
        total_columns.append(rf_columns[0])
        total_rewards.append(rf_rewards[0])
        aaa += 1
        print("保存成功!")
        print("当前为{}次".format(aaa))

    del rf_model
    del rf_columns
    del rf_rewards
    gc.collect()

    '''
    模型选择
    '''
    target_selector = np.array(total_rewards).T
    avg_rewards = pd.Series(target_selector.mean(axis=1), index=train_linear.index, name="reward")

    target_selector = np.argmax(target_selector, axis=1)
    train_linear = pd.concat([train_linear, avg_rewards], axis=1)
    last_reward = train_linear['reward']

    print("training selection model")
    modelselector = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=SEED,
                                                  verbose=0)
    modelselector.fit(train_linear[linear_columns_new + ['reward']], target_selector)
    print(pd.DataFrame(modelselector.feature_importances_, index=linear_columns_new + ['reward']).sort_values(by=[0]).tail( 20))

    for modelp in total_models:
        print("")
        print(modelp)

    lastvalues = train_linear[['id'] + linear_columns_new].copy()
    '''
    最后预测，如果有del,就用gc.collect() 清除内存，防止del没干净。
    '''
    print("end of training, now predicting")
    indice = 0
    countplus = 0
    rewards = []
    infoList = []

    while True:
        infoDict = dict()
        indice += 1
        # test = pd.read_csv(PATH1 + 'test.csv')
        # test = numeric_non_scale(test)                # 类别型数值化
        # test = numeric_scale(test)

        test = pd.read_csv(PATH4+'test_end.csv')
        '''
        test.id.isin 返回boolean DataFrame，显示DataFrame中的每个元素是否包含在值中。
        '''
        indexcommun = list(set(lastvalues.id) & set(test.id))
        print(indexcommun)
        lastvalues = test[test.id.isin(indexcommun)]['id']
        print(lastvalues)



        test['reward'] = last_reward

        selected_prediction = modelselector.predict_proba(test.loc[:, list(test.columns) + ['reward']])
        predict_end = 0
        for ind, elt in enumerate(total_models):
            predict_end += selected_prediction[:, ind] * elt.predict(test[total_columns[ind]])


        last_reward = reward
        rewards.append(reward)
        if reward > 0:
            countplus += 1

        if indice % 100 == 0:
            print(indice, countplus, reward, np.mean(rewards))

        '''
        保存结果
        '''
        test['click'] = selected_prediction
        test[['id', 'click']].to_csv('submission_rl.csv', index=False)








if __name__ == '__main__':
    main()















