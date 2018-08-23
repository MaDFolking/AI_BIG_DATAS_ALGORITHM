
import numpy as np
from sklearn.model_selection import train_test_split

'''
全局函数/变量
'''

PATH  = ''                                                              # 路径
VALIDATION_SIZE = 0.1                                                   # 拆分数据集比例
SEED = 2018                                                             # 随机种子
STOP_ROUNDS = 15                                                        # 多少次不提升就停止的次数
BASE_LINE = ['rmse','auc','mae','error']                                # 评估指标
TRAIN_CHUNK = 10000                                                     # 训练集文件数量
TEST_CHUNK = 10000                                                      # 测试集文件数量
IS_CHUNK = False                                                        # 是否生成TextFileReader对象
loc = locals()                                                          # 获取变量名
plot = True                                                             # 是否开启数据可视化
CV = 10                                                                 # 交叉验证个数
features = ["invited", "user_reco", "evt_p_reco", "evt_c_reco",         # 最后得到的强特征，随时更新
            "user_pop", "frnd_infl", "evt_pop"]

sgd_params = {                                                          # SGD参数
    'loss': "log",
    'penalty': "l2"
}

xgb_params = {                                                          # XGB参数
    'n_estimators': 1000,
    'objective': 'reg:linear',
    'booster': 'gbtree',
    'learning_rate': 0.02,
    'max_depth': 10,
    'max_leaf_nodes': 100,
    'min_child_weight': 10,
    'gamma': 1.45,
    'alpha': 0.0,
    'lambda': 0.1,
    'subsample': 0.67,
    'colsample_bytree': 0.054,
    'colsample_bylevel': 0.50,
    'n_jobs': -1,
    'random_state': SEED
}

fit_params = {                                                          # XGB拟合参数
    'early_stopping_rounds': STOP_ROUNDS,
    'eval_metric': BASE_LINE[1],
    'verbose': True
}

rf_params = {                                                           # RF参数
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 8,
    'max_leaf_nodes': 100,
    'min_impurity_decrease': 0.0001,
    'random_state': SEED,
    'n_jobs': -1
}

lgb_params = {                                                          # LGB参数
    'objective': 'multiclass',
    'num_leaves': 58,
    'subsample': 0.6143,
    'colsample_bytree': 0.6453,
    'min_split_gain': np.power(10, -2.5988),
    'reg_alpha': np.power(10, -2.2887),
    'reg_lambda': np.power(10, 1.7570),
    'min_child_weight': np.power(10, -0.1477),
    'seed': SEED,
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'learning_rate': 0.05,
    'nthread': -1
}

lr_params = {                                                          # LR参数
    'solver': 'sag',
    'class_weight': 'balanced',
    'random_state': SEED,
    'n_jobs': -1
}

'''
拆分数据集
'''
def trans_test_split(*arrays,VALIDATION_SIZE = 0.1,random_state = None):
    return train_test_split(*arrays,test_size = VALIDATION_SIZE,random_state = random_state)
