
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from utils import CV,rmsple,TEST_SIZE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection._split import check_cv
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone


'''
XGB模型构建
依照官网重写XGB函数即可
'''
class XGBClassifyCV(BaseEstimator, RegressorMixin):
    def __init__(self, xgb_params=None, fit_xgb_params=None, cv=CV):
        self.xgb_params = xgb_params
        self.fit_xgb_params = fit_xgb_params
        self.cv = cv

    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis=0)

    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)

    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score
            )
        return np.array(best_scores)

    @property
    def cv_scores_(self):
        return self.scores

    @property
    def cv_score_(self):
        return np.mean(self.scores)

    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration
            )
        return np.array(best_iterations)

    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))

    def fit(self, X, y, **fit_xgb_params):
        cv = check_cv(self.cv, y, classifier=False)
        print(X.shape," ",y.shape)
        self.estimators_ = []
        self.scores = []
        for train, valid in cv.split(X, y):
            model =  xgb.XGBRegressor(**self.xgb_params).fit(
                    X.iloc[train], y.iloc[train],
                    eval_set=[(X.iloc[valid], y.iloc[valid])],
                    **self.fit_xgb_params
                )
            self.estimators_.append(model)
            score = rmsple(y.iloc[valid], model.predict(X.iloc[valid]))
            self.scores.append(score)
        return self

    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)


'''
LGB模型构建
'''
class LGBClassifyCV(BaseEstimator, RegressorMixin):
    def __init__(self, lgb_params=None,lgb_fit_params = None,cv=CV):
        self.lgb_params = lgb_params
        self.lgb_fit_params = lgb_fit_params
        self.cv = cv

    @property
    def cv_scores_(self):
        return self.scores

    @property
    def cv_score_(self):
        return np.mean(self.scores)

    def fit(self, X, y,**lgb_fit_params):
        self.estimators_ = []
        self.scores = []
        for i in range(self.cv):
            x1, x2, y1, y2 = train_test_split(X,y, test_size=TEST_SIZE,random_state=i)
            lg_train = lgb.Dataset(x1, label = y1)
            lg_valid = lgb.Dataset(x2, label = y2)
            model = lgb.train(self.lgb_params,
                              lg_train,
                              num_boost_round = 10000,
                              valid_sets=[lg_valid],
                              **lgb_fit_params)
            self.estimators_.append(model)
            score = rmsple(y2,model.predict(x2,num_iteration=model.best_iteration))
            self.scores.append(score)
        return self

    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)

'''
RF模型构建
'''
class RandomForestClassifyCV(BaseEstimator,RegressorMixin):
    def __init__(self,rf_params = None,cv = CV):
        self.rf_params = rf_params
        self.cv = cv

    @property
    def cv_scores_(self):
        return self.scores

    @property
    def cv_score_(self):
        return np.mean(self.scores)

    def fit(self,X,y):
        print("随机森林开始拟合")
        cv = check_cv(self.cv,y,classifier = False)
        self.estimators_ = []
        self.scores = []
        for train,valid in cv.split(X,y):
            model = RandomForestRegressor(**self.rf_params).fit(X.iloc[train], y.iloc[train])
            self.estimators_.append(model)
            score = rmsple(y.iloc[valid],model.predict(X.iloc[valid]))
            self.scores.append(score)
        return self

    def predict(self,X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred,axis = 0)


'''
模型融合--加权平均值法
'''
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models,weights):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pre_d= np.array([model.predict(X) for model in self.models_])
        for data in range(pre_d.shape[1]):
            single = [pre_d[model, data] * weight for model, weight in zip(range(pre_d.shape[0]), self.weights)]
            w.append(np.sum(single))
        return w
