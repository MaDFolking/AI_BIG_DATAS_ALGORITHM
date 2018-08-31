
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp


from functools import partial
from main_test import main
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import train_test_split
from tf_model  import RegressionHuber, RegressionClf, prelu
from tf_dataset import prepare_vectorizer_1_tf, prepare_vectorizer_3_tf, prepare_vectorizer_2_tf
from util_model import (TEST_SIZE,DUMP_DATASET, USE_CACHED_DATASET, DEBUG_N, HANDLE_TEST, logger, MIN_PRICE_PRED,
                             MAX_PRICE_PRED,Timer,rmsle,PATH,DEBUG,TEST_CHUNK,VALIDATION_SIZE)

'''
模型融合阶段和main函数
'''
'''
2 ** 10 = 1024
2 ** 11 = 2048
我们先用特征1的方式，处理这四个模型。
batch数: 2048
第一种层为 192 64 32
第二种 192 64
'''
def define_models_1(n_jobs, seed):
    h0 = 192  # the same to make training take the same time
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu)
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_1_tf(n_jobs=n_jobs)

'''
用特征2的处理方式，处理四个模型,里面并且融入了二值化处理.
batch数: 5096
第一种层为 192 64 32
第二种 192 64
'''
def define_models_2(n_jobs, seed):
    h0 = 192  # the same to make training take the same time
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu, binary_X=True),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu, binary_X=True),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu)
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_2_tf(n_jobs=n_jobs)

'''
用特征3方式生成词向量，处理。
batch数: 2048
部分进行二值化处理
第一种层为 128 64 32
第二种 128 64
'''
def define_models_3(n_jobs, seed):
    h0 = 128  # reduced from 192 due to kaggle slowdown
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu, binary_X=True),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu, binary_X=True),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu),
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_3_tf(n_jobs)

'''
每种方法，都有个main方法，可以比较，但我们最终是用强化学习进行融合处理。
'''

'''
sys.argv[] http://www.cnblogs.com/aland-1415/p/6613449.html
说白了就是一个从程序外部获取参数的桥梁，这个“外部”很关键，
所以那些试图从代码来说明它作用的解释一直没看明白。因为我们从外部取得的参数可以是多个,
所以获得的是一个列表（list)，也就是说sys.argv其实可以看作是一个列表，所以才能用[]提取其中的元素。其第一个元素是程序本身，随后才依次是外部给予的参数。
'''
'''
需要用python命令开启，可以在1,2,3三个数中选择一个。
'''

if __name__ == '__main__':
    main(
        'tf',
        '1',                                    #相当于获取第二个参数
        {
            1:define_models_1(n_jobs = 4,seed = 1),
            2:define_models_2(n_jobs = 4,seed = 2),
            3:define_models_3(n_jobs = 4,seed = 3)
        }
    )
