import gc
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score





SEED = 2018                                                    # 随机种子
CV = 10                                                        # 交叉验证数
PATH = ''                                                      # 路径
NUM_FEATURE = 800                                              # 通过PCA主成分分析，大概有993个特征，
                                                               # 但那是增加特征后的，并没处理的，后来测试500-900之间效果都可以，这里取800
NUM_OF_COM = 10                                                # SparseRandom的随机映射维度，增加它可以健壮模型，防止过拟合。
TEST_SIZE = 0.2                                                # 数据集拆分8:2

'''
XGB参数
'''
xgb_params =  {
        'n_estimators': 1000,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 10,
        'max_leaf_nodes' : 100,
        'min_child_weight': 10,
        'gamma': 1.45,
        'alpha': 0.0,
        'lambda': 0.1,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 2018
    }
'''
XGB拟合参数
'''
fit_xgb_params = {
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse',
        'verbose': True
    }

'''
LGB参数
'''
lgb_params = {
    'objective': 'regression',
    'num_leaves': 58,
    'subsample': 0.6143,
    'colsample_bytree': 0.6453,
    'min_split_gain': np.power(10, -2.5988),
    'reg_alpha': np.power(10, -2.2887),
    'reg_lambda': np.power(10, 1.7570),
    'min_child_weight': np.power(10, -0.1477),
    'verbose': -1,
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'learning_rate': 0.05,
    'metric': 'l2',
    'n_jobs': -1
}

'''
LGB拟合参数
'''
lgb_fit_params = {
    'num_boost_round':1000,
    'early_stopping_rounds':50,
    'verbose_eval' : 2
}


'''
RF参数
'''
rf_params = {
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 13,
    'max_leaf_nodes': 100,
    'min_impurity_decrease': 0.0001,
    'random_state': SEED,
    'verbose':2,
    'n_jobs': -1
}

'''
评估器
'''
def rmsple(y_true, y_pre):
    return np.sqrt(np.square(y_true - y_pre).mean())

def rmse(y_true, y_pre):
    return mean_squared_error(y_true, y_pre) ** .5

'''
交叉验证
'''
def rmse_cv(model,X_train,y_train):
    kf = KFold(CV,shuffle = True,random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

'''
加载数据
'''
def load_data():
    X_train = pd.read_csv(PATH + 'train.csv')
    X_test = pd.read_csv(PATH + 'test.csv')
    y_train = np.log1p(X_train['target'])
    X_Id = X_test['ID']

    del X_train['ID']
    del X_train['target']
    del X_test['ID']
    gc.collect()
    return X_train,X_test,y_train,X_Id.values

'''
去除方差/标准差为0的特征
'''
def drop_variance(X):
    colsToRemove = []
    columns = X.columns
    for i in columns:
        if X[i].std() == 0:
            colsToRemove.append(i)
    return colsToRemove

'''
去除相邻俩个线性相同的特征
'''
def drop_linear_same(X):
    colsToRemove = []
    columns = X.columns
    for i in range(len(columns)-1):
        vals = X[columns[i]].values
        for j in range(i+1,len(columns)):
            if np.array_equal(vals,X[columns[j]]):
                colsToRemove.append(columns[j])
    if len(colsToRemove) > 0:
        print("线性相同的特征为: {}".format(colsToRemove))
        return colsToRemove
    else:
        print("没有线性相同的特征")
        return colsToRemove

'''
对方差小的且不为0的数进行标准化，目的是使得结果更平滑，方便我们拟合更好的模型。
'''
def drop_feature_same_value(X):
    columns = X.columns
    for col in columns:
        data = X[col].values
        data_mean, data_std = np.mean(data), np.std(data)
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        outList = [i for i in data if i > lower and i < upper]
        if (len(outList) > 0):
            non_zero_idx = data != 0
            X.loc[non_zero_idx, col] = np.log(data[non_zero_idx])
        nonzero_rows = X[col] != 0
        '''
        防止元素异常，我们用isfinite测试
        '''
        if np.isfinite(X.loc[nonzero_rows, col]).all():
            X.loc[nonzero_rows, col] = scale(X.loc[nonzero_rows, col])
            if np.isfinite(X[col]).all():
                X[col] = scale(X[col])
        return X

'''
去噪:将用户交易额小于10元的设置为False
'''
def has_ugly(row):
    for v in row.values[row.values > 0]:
        if str(v)[::-1].find('.') > 2:
            return True
    return False

'''
XGB提取特征
'''
def FeatureXgbHandle(X,y):
    print("XGB提取特征开始...")
    reg = XGBRegressor(n_estimators=1000, n_jobs=-1, nthread=-1)
    folds = KFold(4, True, 134259)
    fold_idx = [(trn_, val_) for trn_, val_ in folds.split(X)]
    scores = []
    features = [f for f in X.columns if f not in ['f190486d6']]

    for _f in features:
        score = 0
        for trn_, val_ in fold_idx:
            reg.fit(
                X[['f190486d6', _f]].iloc[trn_], y.iloc[trn_],
                eval_set=[(X[['f190486d6', _f]].iloc[val_], y.iloc[val_])],
                eval_metric='rmse',
                early_stopping_rounds=15,
                verbose=True
            )
            score += rmse(y.iloc[val_], reg.predict(X[['f190486d6', _f]].iloc[val_],
                                                         ntree_limit=reg.best_ntree_limit)) / folds.n_splits
        scores.append((_f, score))

    report = pd.DataFrame(scores, columns=['feature', 'rmse']).set_index('feature')
    report.sort_values(by='rmse', ascending=True, inplace=True)
    report.to_csv(PATH + 'feature_report_xgb_123.csv', index=True)
