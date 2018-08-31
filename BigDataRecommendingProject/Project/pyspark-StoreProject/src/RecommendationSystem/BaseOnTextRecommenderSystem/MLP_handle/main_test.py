
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp


from functools import partial
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import train_test_split
from tf_model  import RegressionHuber, RegressionClf, prelu
from tf_dataset import prepare_vectorizer_1_tf, prepare_vectorizer_3_tf, prepare_vectorizer_2_tf
from util_model import (TEST_SIZE,DUMP_DATASET, USE_CACHED_DATASET, DEBUG_N, HANDLE_TEST, logger, MIN_PRICE_PRED,
                             MAX_PRICE_PRED,Timer,rmsle,PATH,DEBUG,TEST_CHUNK,VALIDATION_SIZE)

'''
main函数阶段，我们后面所有模型都调用这个使用，最后再用强化学习总结处理
'''


'''
下面是模型拟合和main函数
'''

'''
拟合一个模型，取最大最小值
'''
def fit_one(est, X, y):
    print("fitting y min={} max={}".format(y.min(), y.max()))
    return est.fit(X, y)


'''
预测一个模型
'''
def predict_one(est, X):
    yhat = est.predict(X)
    print("predicting y min={} max={}".format(yhat.min(), yhat.max()))
    return yhat


'''
预测模型
'''
def predict_models(X, fitted_models, vectorizer=None, parallel='thread'):
    if vectorizer:
        with Timer('Transforming data'):
            X = vectorizer.transform(X)
    predict_one_ = partial(predict_one, X=X)
    preds = map_parallel(predict_one_, fitted_models, parallel)
    return np.expm1(np.vstack(preds).T)

'''
拟合模型
'''
def fit_models(X_tr, y_tr, models, parallel='thread'):
    y_tr = np.log1p(y_tr)
    fit_one_ = partial(fit_one, X=X_tr, y=y_tr)
    return map_parallel(fit_one_, models, parallel)

'''
并行处理
'''

'''
由于GIL（全局解释锁）的问题，python多线程并不能充分利用多核处理器。如果想要充分地使用多核CPU的资源，
在python中大部分情况需要使用多进程。multiprocessing可以给每个进程赋予单独的Python解释器，这样就规避了全局解释锁所带来的问题。
与threading.Thread类似，可以利用multiprocessing.Process对象来创建一个进程。multiprocessing支持子进程、通信和共享数据、执行不同形式的同步，
提供了Process、Queue、Pipe、Lock等组件。
'''

def map_parallel(fn, lst, parallel, max_processes=4):
    if parallel == 'thread':
        with ThreadPool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel == 'mp':
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel is None:
        return list(map(fn, lst))
    else:
        raise ValueError('并行异常为: {}'.format(parallel))

'''
预测测试集一次性迭代数
'''
def predict_models_test_batches(models, vectorizer, parallel='thread'):
    chunk_preds = []
    test_idx = []
    for df in load_test_iter():
        test_idx.append(df.test_id.values)
        print("预测的迭代最小值为: {} , 最大值为: {}".format(df.test_id.min(), df.test_id.max()))
        chunk_preds.append(predict_models(df, models, vectorizer=vectorizer, parallel=parallel))
    predictions = np.vstack(chunk_preds)
    test_idx = np.concatenate(test_idx)
    return test_idx, predictions

'''
生成提交文件
'''
def make_submission(te_idx, preds, save_as):
    submission = pd.DataFrame({
        "test_id": te_idx,
        "price": preds
    }, columns=['test_id', 'price'])
    submission.to_csv(save_as, index=False)

'''
转化为向量，并直接拆分。
'''
def fit_transform_vectorizer(vectorizer):
    df_tr, df_va = load_train_validation()
    y_tr = df_tr.price.values
    y_va = df_va.price.values
    X_tr = vectorizer.fit_transform(df_tr, y_tr)
    X_va = vectorizer.transform(df_va)
    return X_tr, y_tr, X_va, y_va, vectorizer

'''
拟合验证集
'''
def fit_validate(models, vectorizer, name=None,
                 fit_parallel='thread', predict_parallel='thread'):
    cached_path = 'data_{}.pkl'.format(name)
    if USE_CACHED_DATASET:
        assert name is not None
        with open(cached_path, 'rb') as f:
            X_tr, y_tr, X_va, y_va, fitted_vectorizer = pickle.load(f)
        if DEBUG_N:
            X_tr, y_tr = X_tr[:DEBUG_N], y_tr[:DEBUG_N]
    else:
        X_tr, y_tr, X_va, y_va, fitted_vectorizer = fit_transform_vectorizer(vectorizer)
    if DUMP_DATASET:
        assert name is not None
        with open(cached_path, 'wb') as f:
            pickle.dump((X_tr, y_tr, X_va, y_va, fitted_vectorizer), f)
    fitted_models = fit_models(X_tr, y_tr, models, parallel=fit_parallel)
    y_va_preds = predict_models(X_va, fitted_models, parallel=predict_parallel)
    return fitted_vectorizer, fitted_models, y_va, y_va_preds

'''
合并最终的预测值，以便提高防止过拟合能力，最后需要一个线性回归得到权重，我们又需要特征和稀疏矩阵，所以用生成稀疏矩阵的Lasso回归。
'''
def merge_predictions(X_tr, y_tr, X_te=None, est=None, verbose=True):
    if est is None:
        est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')
    est.fit(np.log1p(X_tr), np.log1p(y_tr))
    if hasattr(est, 'intercept_') and verbose:
        logger.info('merge_predictions = \n{:+.4f}\n{}'.format(
            est.intercept_,
            '\n'.join('{:+.4f} * {}'.format(coef, i) for i, coef in zip(range(X_tr.shape[0]), est.coef_))))
    return (np.expm1(est.predict(np.log1p(X_tr))),np.expm1(est.predict(np.log1p(X_te))) if X_te is not None else None)


'''
加载训练数据集数据
DEBUG，大于0的数。
'''
def load_train():
    if DEBUG:
        return pd.read_csv(PATH+'train.tsv', sep='\t').query('price > 0').iloc[:DEBUG_N, :]
    else:
        return pd.read_csv(PATH+'train.tsv', sep='\t').query('price > 0')

'''
拆分数据集
'''
def load_train_validation():
    return mercari_train_test_split(load_train())

'''
加载测试数据集，并直接拆分
'''
def load_test_iter():
    for _ in range(TEST_SIZE):
        for df in pd.read_csv(PATH+'test.tsv', sep='\t', chunksize=TEST_CHUNK):
            if DEBUG:
                yield df.iloc[:DEBUG_N]
            else:
                yield df

'''
VALIDATION_SIZE 默认为0.05进行拆分
'''
def mercari_train_test_split(*arrays):
    return train_test_split(*arrays, test_size=VALIDATION_SIZE, random_state=0)

'''
加载数据
'''
def load_data():
    print("加载数据开始...")
    train = pd.read_csv(PATH+'train.tsv',sep = '\t')
    test = pd.read_csv(PATH+'test.tsv',sep = '\t')
    target = np.log1p(train.price)
    train = train[train['price']>0]
    if train is None or test is None or target is None:
        raise IOError("加载数据失败!")
    print("加载数据结束...\n\n{}\n".format("*"*200))
    return train,test,target

'''
数据探索
'''
def explore_data(train,test,target):
    print("数据探索开始...")
    if train is None or test is None or target is None:
        raise ValueError("输入的训练集不能为空!")
    print('训练集样本: \n{}\n测试集样本: \n{}\n目标样本: \n{}\n{}'
          .format(train.head(),test.head(),target.head(),'*'*50))
    print('训练集样本具体描述: \n{}\n测试集样本具体描述: \n{}\n目标样本具体描述: \n{}\n{}'
          .format(train.describe(), test.describe(), target.describe(),'*'*50))
    print('训练集样本维度: \n{}\n测试集样本维度: \n{}\n目标样本维度: \n{}\n{}'
          .format(train.shape, test.shape, target.shape, '*'*50))
    print("数据探索结束...\n\n{}\n".format("*" * 200))

'''
main里处理模型融合
'''
def main(name, action, arg_map, fit_parallel='thread', predict_parallel='thread'):
    # train, test, target = load_data()
    # explore_data(train, test, target)

    prefix = lambda r: '{}_{}s'.format(name, r)

    if action in ("1", "2", "3"):
        model_round = int(action)
        models, vectorizer = arg_map[model_round]
        vectorizer, fitted_models, y_va, y_va_preds = fit_validate(
            models, vectorizer, name = model_round,
            fit_parallel=fit_parallel, predict_parallel=predict_parallel)
        joblib.dump(y_va_preds, "{}_va_preds.pkl".format(prefix(model_round)), compress=3)
        if HANDLE_TEST:
            test_idx, y_te_preds = predict_models_test_batches(
                fitted_models, vectorizer, parallel=predict_parallel)
            joblib.dump(y_te_preds, "{}_te_preds.pkl".format(prefix(model_round)), compress=3)
            joblib.dump(test_idx, "test_idx.pkl", compress=3)
        joblib.dump(y_va, "y_va.pkl", compress=3)
        for i in range(y_va_preds.shape[1]):
            print("Model {} rmsle {:.4f}".format(i, rmsle(y_va_preds[:, i], y_va)))
        print("Model mean rmsle {:.4f}".format(rmsle(y_va_preds.mean(axis=1), y_va)))

    elif action == "merge":
        va_preds = []
        te_preds = []
        for model_round in ("1", "2", "3"):
            try:
                va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
                if HANDLE_TEST:
                    te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
            except Exception as e:
                print('Warning: error loading round {model_round}: {e}')
                traceback.print_exc()
        va_preds = np.hstack(va_preds).clip(MIN_PRICE_PRED, MAX_PRICE_PRED)
        if HANDLE_TEST:
            te_preds = np.hstack(te_preds).clip(MIN_PRICE_PRED, MAX_PRICE_PRED)
        else:
            te_preds = None
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        print("Stacking rmsle", rmsle(y_va, va_preds_merged))
        if HANDLE_TEST:
            test_idx = joblib.load("test_idx.pkl")
            make_submission(test_idx, te_preds_merged, 'submission_merged.csv')

    elif action == "merge_describe":
        va_preds = []
        te_preds = []
        for model_round in ("1", "2", "3"):
            va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
            te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
        va_preds = np.hstack(va_preds)
        te_preds = np.hstack(te_preds)
        _, df_va = load_train_validation()
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        print("Stacking rmsle", rmsle(y_va, va_preds_merged))
        df_va['preds'] = va_preds_merged
        df_va['err'] = (np.log1p(df_va['preds']) - np.log1p(df_va['price'])) ** 2
        df_va.sort_values('err', ascending=False).to_csv('validation_preds.csv', index=False)

