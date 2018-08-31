'''
这次，我不会对这个项目进行测试，这里是构建这次我们要用的常用模型
'''

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from matplotlib import pylab as plt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._split import check_cv,KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone


'''
XGB模型构建
依照官网重写XGB函数即可
'''
class XGBClassifyCV(BaseEstimator, RegressorMixin):
    def __init__(self, xgb_params=None, fit_params=None, cv=CV):
        self.xgb_params = xgb_params
        self.fit_params = fit_params
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
        return self.best_scores_

    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)

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

    def fit(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=True)
        self.estimators_ = []

        for train, valid in cv.split(X, y):
            self.estimators_.append(
                xgb.XGBClassifier(**self.xgb_params).fit(
                    X[train], y[train],
                    eval_set=[(X[valid], y[valid])],
                    **self.fit_params
                )
            )

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
    def __init__(self, lgb_params=None,cv=CV):
        self.lgb_params = lgb_params
        self.cv = cv

    @property
    def cv_scores_(self):
        return self.scores

    @property
    def cv_score_(self):
        return self.score

    def fit(self, X, y):
        cv = check_cv(self.cv, y, classifier=True)
        self.estimators_ = []
        self.scores = []
        self.score = 0
        for train, valid in cv.split(X, y):
            score1 = 0
            test = len(y[valid])
            print(X[train].shape)
            print(y[train].shape)
            clf =  lgb.LGBMClassifier(**self.lgb_params).fit( X[train], y[train],eval_set=[(X[train],y[train])],early_stopping_rounds = 15)
            for i in range(0, test):
                yt = clf.predict(X[valid][i, :])
                if yt == y[valid][i]:
                    score1 += 1
            score1 = score1 / test
            print(score1)
            self.scores.append(score1)
            self.estimators_.append(clf)
        self.score = sum(self.scores) / len(self.scores)
        return self

    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)



'''
SGD模型构建
'''
class SGDClassifyCV(BaseEstimator, RegressorMixin):
    def __init__(self,sgd_params = None,cv = CV):
        self.sgd_params = sgd_params
        self.cv = cv

    @property
    def cv_scores_(self):
        return self.scores

    @property
    def cv_score_(self):
        return self.score


    def fit(self,X,y):
        cv = check_cv(self.cv,y,classifier = True)
        self.estimators_ = []
        self.scores = []
        self.score = 0
        for train,valid in cv.split(X,y):
            score1 = 0
            test = len(y[valid])
            clf = SGDClassifier(**self.sgd_params).fit(X[train],y[train])
            for i in range(0,test):
                yt = clf.predict(X[valid][i,:])
                if yt == y[valid][i]:
                    score1 += 1
            score1 = score1 / test
            print(score1)
            self.scores.append(score1)
            self.estimators_.append(clf)
        self.score = sum(self.scores) / len(self.scores)
        return self

    def predict(self,X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred,axis = 0)


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
        return self.score

    def fit(self,X,y):
        cv = check_cv(self.cv,y,classifier = True)
        self.scores = []
        self.score = 0
        self.estimators_ = []
        for train,valid in cv.split(X,y):
            score1 = 0
            test = len(y[valid])
            print("随机森林开始拟合")
            clf =RandomForestClassifier(**self.rf_params).fit(X[train], y[train])
            print("随机森林拟合结束")
            for i in range(0, test):
                yt = clf.predict(X[valid][i, :])
                if yt == y[valid][i]:
                    score1 += 1
            score1 = score1 / test
            print(score1)
            self.scores.append(score1)
            self.estimators_.append(clf)
        self.score = sum(self.scores) / len(self.scores)
        return self

    def predict(self,X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred,axis = 0)

'''
LR模型构建
'''
class LRClassifyCV(BaseEstimator,RegressorMixin):
    def __init__(self,lr_params = None,cv = CV):
        self.lr_params = lr_params
        self.cv = cv

    @property
    def cv_scores_(self):
        return self.scores

    @property
    def cv_score_(self):
        return self.score

    def fit(self,X,y):
        cv = check_cv(self.cv,y,classifier = True)
        self.scores = []
        self.score = 0
        self.estimators_ = []
        for train,valid in cv.split(X,y):
            score1 = 0
            test = len(y[valid])
            print("逻辑回归开始拟合")
            clf =LogisticRegression(**self.lr_params).fit(X[train], y[train])
            print("逻辑回归拟合结束")
            # for i in range(0, test):
            #     yt = clf.predict(X[valid][i,:])
            #     if yt == y[valid][i]:
            #         score1 += 1
            # score1 = score1 / test
            # print(score1)
            # self.scores.append(score1)
            self.estimators_.append(clf)
        #self.score = sum(self.scores) / len(self.scores)
        return self

    def predict(self,X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred,axis = 0)

'''
模型融合--stacking
'''
class StackingModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean
