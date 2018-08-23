


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





def main():
    DataCleaner()

    lgb_params = {
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


    lgb_model = LGBClassifyCV(lgb_params = lgb_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                               columns=features))
    test_X = np.matrix(pd.DataFrame(testDf,index = None,columns = features))
    y = trainDf.interested
    lgb_model.fit(train_X,y)

    print("LGB模型最优参数为:",lgb_model.get_params)
    print("LGB模型Auc值为:",lgb_model.cv_scores_)
    print("LGB模型最佳Auc值为:",lgb_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('LgbModel1.pickle', 'wb') as writer:
            pickle.dump(lgb_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))

    print("模型开始读取...")
    with open('LgbModel1.pickle', 'rb') as reader:
        lgb_model = pickle.load(reader)

    print("模型读取结束...\n\n{}\n".format("*" * 200))

    programEntities = UserOfEventCleaner()
    UserSimilar(programEntities = programEntities)
    EventSimilar(programEntities = programEntities)
    UserFriends(programEntities = programEntities)
    EventAttend(programEntities = programEntities)

    dr = FeatureEngineering()
    print("生成训练数据...\n")
    dr.rewriteData(train=True, start=2, header=True)
    print("生成预测数据...\n")
    dr.rewriteData(train=False, start=2, header=True)
    xgb_params = {
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
        'random_state': SEED
    }

    fit_params = {
        'early_stopping_rounds': STOP_ROUNDS,
        'eval_metric': BASE_LINE[1],
        'verbose': True
    }
    xgb_model = XGBClassifyCV(xgb_params = xgb_params,fit_params = fit_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                               columns=features))
    test_X = np.matrix(pd.DataFrame(testDf,index = None,columns = features))
    y = np.array(trainDf.interested)
    xgb_model.fit(train_X,y)

    print("XGB模型最优参数为:",xgb_model.get_params)
    print("XGB模型Auc值为:",xgb_model.cv_scores_)
    print("XGB模型最佳Auc值为:",xgb_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('XgbModel1.pickle', 'wb') as writer:
            pickle.dump(xgb_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))

    print("模型开始读取...")
    with open('XgbModel1.pickle', 'rb') as reader:
        xgb_model = pickle.load(reader)

    print("模型读取结束...\n\n{}\n".format("*" * 200))

    print("预测结果开始...")
    predict = xgb_model.predict(test_X)
    print(predict)
    print("预测结果结束...\n\n{}\n".format("*" * 200))

    print("可视化特征开始...")
    if plot:
        with open('xgb.fmap', 'w') as writer:
            i = 0
            for feat in features:
                writer.write('{0}\t{1}\tq\n'.format(i, feat))
                i = i + 1
        importance = xgb_model.feature_importances_
        print(importance)
        importance = zip(features,importance)
        importance = sorted(importance)
        df = pd.DataFrame(importance, columns=['feature', 'score'])
        df['score'] = df['score'] / df['score'].sum()
        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(25, 15))
        plt.title('XGBoost_Feature_Importance')
        plt.xlabel('Feature_Importance')
        plt.gcf().savefig('推荐系统—特征重要程度可视化-XGBoost.png')
    print("可视化特征结束...\n\n{}\n".format("*" * 200))

    sgd_params = {
        'loss':"log",
        'penalty':"l2"
    }
    sgd_model = SGDClassifyCV(sgd_params = sgd_params)

    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0,inplace = True)
    testDf.fillna(0, inplace = True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    #test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)
    sgd_model.fit(train_X, y)
    print("SGD模型最优参数为:", sgd_model.get_params)
    print("SGD模型Auc值为:", sgd_model.cv_scores_)
    print("SGD模型最佳Auc值为:", sgd_model.cv_score_)

    print("模型开始保存...")
    try:
        with open('SgdModel1.pickle', 'wb') as writer:
            pickle.dump(sgd_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))
    print("模型开始读取...")
    with open('SgdModel1.pickle', 'rb') as reader:
        sgd_model = pickle.load(reader)
    print("模型读取结束...\n\n{}\n".format("*" * 200))

    rf_params = {
        'n_estimators':100,
        'max_features':0.5,
        'max_depth':8,
        'max_leaf_nodes':100,
        'min_impurity_decrease':0.0001,
        'random_state':SEED,
        'n_jobs':-1
    }
    rf_model = RandomForestClassifyCV(rf_params = rf_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    # test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)
    rf_model.fit(train_X, y)
    print("随机森林模型最优参数为:", rf_model.get_params)
    print("随机森林模型Auc值为:", rf_model.cv_scores_)
    print("随机森林模型最佳Auc值为:", rf_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('RfModel1.pickle', 'wb') as writer:
            pickle.dump(rf_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))
    print("模型开始读取...")
    with open('RfModel1.pickle', 'rb') as reader:
        rf_model = pickle.load(reader)
    print("模型读取结束...\n\n{}\n".format("*" * 200))
    lr_params = {
        'solver':'sag',
        'class_weight':'balanced',
        'random_state':SEED,
        'n_jobs':-1
    }
    lr_model = LRClassifyCV(lr_params = lr_params)
    lr_model.fit(train_X, y)
    print("逻辑回归模型最优参数为:", lr_model.get_params)
    print("逻辑回归模型Auc值为:", lr_model.cv_scores_)
    print("逻辑回归模型最佳Auc值为:", lr_model.cv_score_)
    print("模型开始保存...")
    try:
        with open('LrModel1.pickle', 'wb') as writer:
            pickle.dump(lr_model, writer)
    except IOError:
        print("模型保存失败...")
    print("模型保存结束...\n\n{}\n".format("*" * 200))
    print("模型开始读取...")
    with open('LrModel1.pickle', 'rb') as reader:
        lr_model = pickle.load(reader)
    print("模型读取结束...\n\n{}\n".format("*" * 200))

    '''
    模型融合
    '''
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    # test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)

    '''
    模型加载
    '''
    with open('LrModel1.pickle','rb') as reader_lr:
        lr_model = pickle.load(reader_lr)
    # with open('RfModel1.pickle','rb') as reader_rf:
    #     rf_model = pickle.load(reader_rf)
    with open('XgbModel1.pickle','rb') as reader_xgb:
        xgb_model = pickle.load(reader_xgb)
    with open('SgdModel1.pickle','rb') as reader_sgd:
        sgd_model = pickle.load(reader_sgd)

    # print("逻辑回归模型最优参数为:", lr_model.get_params)
    # print("逻辑回归模型Auc值为:", lr_model.cv_scores_)
    # print("逻辑回归模型最佳Auc值为:", lr_model.cv_score_)

    print("开始模型融合...")
    total_model = StackingModel(mod=(xgb_model,sgd_model),meta_model = lr_model)
    print("模型融合初始化结束...")
    cv = check_cv(CV, y, classifier=True)
    scores = []
    score = 0
    estimators_ = []
    for train, valid in cv.split(train_X, y):
        score1 = 0
        test = len(y[valid])
        print("融合模型开始拟合")
        clf = total_model.fit(train_X[train],y[train])
        print("融合模型拟合结束")
        for i in range(0, test):
            yt = clf.predict(train_X[valid][i, :])
            if yt == y[valid][i]:
                score1 += 1
        score1 = score1 / test
        print(score1)
        scores.append(score1)
        estimators_.append(clf)

    score = sum(scores) / len(scores)
    print("模型融合的最佳AUC值为:",score)




if __name__ == '__main__':
    main()
