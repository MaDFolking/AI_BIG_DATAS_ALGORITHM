import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn import random_projection
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from utils import drop_variance,drop_linear_same,drop_feature_same_value,NUM_FEATURE,SEED,PATH

'''
数据处理
'''
class DataHandle(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self,isData = True):
        print("数据处理初始化开始...")
        if isData == False:
            raise ImportError("请改成True进行数据处理")
        print("数据处理初始化结束...\n\n{}\n".format("*"*200))

    def fit(self,X,y = None):
        print("数据处理开始...")
        print(X.shape)
        return self

    def transform(self,X):
        drop_cols1 = drop_variance(X)
        X.drop(drop_cols1,axis = 1,inplace = True)

        drop_cols2 = drop_linear_same(X)
        X.drop(drop_cols2,axis = 1,inplace = True)

        X = drop_feature_same_value(X)
        X.fillna(0,inplace = True)

        print("数据处理结束...\n\n{}\n".format("*"*200))
        return X


'''
特征提取
'''
class FeatureExtraction(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self,isData=True):
        print("特征提取初始化开始...")
        if isData == False:
            raise ImportError("请改成True进行特征提取")
        self.models = [GradientBoostingRegressor(random_state = SEED),RandomForestRegressor(random_state = SEED),
                       XGBRegressor(random_state = SEED),lgb.LGBMRegressor(seed = SEED,max_bin = 55)]
        print("特征提取初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, y=None):
        print("特征提取开始...")
        print(X.shape)
        try:
            self.X_new = pd.read_csv(PATH + 'feature_importance_tree_end.csv')
            return self
        except :
            self.X_new = None
            self.feature_importance = []
            '''
            如果想更精确，可以做交叉验证。
            '''
            for clf in self.models:
                clf.fit(X, y)
                features = pd.Series(clf.feature_importances_, index=X.columns)
                self.feature_importance.append(features.nlargest(NUM_FEATURE).index)
            return self

    def transform(self, X):
        if self.X_new is not None:
            columns = self.X_new.columns
            return X[columns]
        else:
            common_feature1 = pd.Series(list(set(self.feature_importance[0]).intersection(set(self.feature_importance[1])))).values
            common_feature2 = pd.Series(list(set(self.feature_importance[2]).intersection(set(self.feature_importance[3])))).values
            common_feature3 = pd.Series(list(set(common_feature1).intersection(set(common_feature2)))).values
            print(common_feature3)
            print("特征提取结束...\n\n{}\n".format("*" * 200))
            X = X[common_feature3]
            X.to_csv(PATH + 'feature_importance_tree_end.csv',index = False)
            return X[common_feature3]

'''
特征挖掘
'''
class FeatureMining(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self,isData=True):
        print("特征挖掘初始化开始...")
        if isData == False:
            raise ImportError("请改成True进行特征挖掘")
        print("特征挖掘初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, y=None):
        print("特征挖掘开始...")
        X.fillna(0,inplace = True)
        corr = X.corr().values
        len_columns = len(X.columns)
        self.corr_list = []
        for i in range(0,len_columns):
            list_columns = []
            for j in range(0,len_columns):
                if corr[i][j] != 1.0 and corr[i][j] > 0.6:
                    list_columns.append(X.columns[i])
                    list_columns.append(X.columns[j])
                    self.corr_list.append(list_columns)
                    print("{}与{}的相关性为{}".format(X.columns[i],X.columns[j],corr[i][j]))
        return self

    def transform(self, X):
        for col in self.corr_list:
            X['{}'.format(col[0])+'_'+'{}'.format(col[1])] = np.add(X[col[0]].values,X[col[1]].values)
        print("特征挖掘结束...\n\n{}\n".format("*" * 200))
        return X



'''
特征构造
'''
class FeatureCreate(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, isData=True):
        print("特征构造初始化开始...")
        if isData == False:
            raise ImportError("请改成True进行特征构造")
        print("特征构造初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, y=None):
        print("特征构造开始...")
        return self

    def transform(self, X):
        X.fillna(0, inplace=True)
        X['log_of_mean'] = np.log1p(X.mean(axis=1))
        X['log_of__sum'] = np.log1p(X.sum(axis=1))
        X['log_of__std'] = np.log1p(X.std(axis=1))
        X['the_kur'] = X.kurtosis(axis=1)
        X['the_skew'] = X.skew(axis=1)
        X['log_of__var'] = np.log1p(X.var(axis=1))
        X['log_of__max'] = np.log1p(X.max(axis=1))
        X['sum_zero'] = np.log1p(X.apply(lambda x: len(x[x == 0]), axis=1))
        X['not_zero_max'] = np.log1p(X.apply(lambda x: x[x != 0].max(), axis=1))
        X['not_zero_min'] = np.log1p(X.apply(lambda x: x[x != 0].min(), axis=1))
        X.fillna(0, inplace=True)

        report = pd.read_csv(PATH + 'feature_report_xgb_123.csv')
        good_features = report.loc[report['rmse'] <= 1.6]['feature']
        good_features = pd.Series(list(set(good_features).intersection(set(X.columns))))

        X['log_of_mean_good'] = np.log1p(X[good_features].mean(axis=1))
        X['log_of__sum_good'] = np.log1p(X[good_features].sum(axis=1))
        X['log_of__std_good'] = np.log1p(X[good_features].std(axis=1))
        X['the_kur_good'] = X[good_features].kurtosis(axis=1)
        X['the_skew_good'] = X[good_features].skew(axis=1)
        X['log_of__var_good'] = np.log1p(X[good_features].var(axis=1))
        X['log_of__max_good'] = np.log1p(X[good_features].max(axis=1))

        transformer = random_projection.SparseRandomProjection(n_components=10)
        transformer.fit(X)
        RP = transformer.transform(X)

        rp = pd.DataFrame(RP)
        columns = ["RandomProjection{}".format(i) for i in range(10)]
        rp.columns = columns
        rp.index = X.index
        X = pd.concat([X, rp], axis=1)
        X.to_csv('feature_handle_train.csv',index = False)
        X.fillna(0, inplace=True)
        print("特征构造结束...\n\n{}\n".format("*" * 200))
        return X



