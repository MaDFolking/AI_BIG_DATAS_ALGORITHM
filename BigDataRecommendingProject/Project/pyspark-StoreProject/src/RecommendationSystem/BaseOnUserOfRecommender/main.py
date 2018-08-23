

import pickle
import numpy as np
import pandas as pd

from matplotlib import pylab as plt
from sklearn.model_selection._split import check_cv
from utils_ml import SEED,plot,CV,features,sgd_params,xgb_params, fit_params,rf_params,lgb_params
from Model_ import XGBClassifyCV,LGBClassifyCV,SGDClassifyCV,RandomForestClassifyCV,LRClassifyCV,StackingModel
from feature_ import DataCleaner,UserFriends,UserOfEventCleaner,EventAttend,EventSimilar,FeatureEngineering,UserSimilar


def main():
    DataCleaner()
    programEntities = UserOfEventCleaner()
    UserSimilar(programEntities = programEntities)
    EventSimilar(programEntities = programEntities)
    UserFriends(programEntities = programEntities)
    EventAttend(programEntities = programEntities)

    dr = FeatureEngineering()
    dr.rewriteData(train=True, start=2, header=True)
    dr.rewriteData(train=False, start=2, header=True)

    '''
    LGB模型
    '''
    lgb_model = LGBClassifyCV(lgb_params = lgb_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")

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

    '''
    XGB模型
    '''
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

    '''
    SGD模型
    '''
    sgd_model = SGDClassifyCV(sgd_params = sgd_params)

    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0,inplace = True)
    testDf.fillna(0, inplace = True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
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

    '''
    RF模型
    '''
    rf_model = RandomForestClassifyCV(rf_params = rf_params)
    trainDf = pd.read_csv("data_train.csv")
    testDf = pd.read_csv("data_test.csv")
    trainDf.fillna(0, inplace=True)
    testDf.fillna(0, inplace=True)
    train_X = np.matrix(pd.DataFrame(trainDf, index=None,
                                     columns=features))
    test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
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

    '''
    LR模型
    '''
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
    test_X = np.matrix(pd.DataFrame(testDf, index=None, columns=features))
    y = np.array(trainDf.interested)

    '''
    模型加载
    '''
    with open('LrModel1.pickle','rb') as reader_lr:
        lr_model = pickle.load(reader_lr)
    with open('RfModel1.pickle','rb') as reader_rf:
        rf_model = pickle.load(reader_rf)
    with open('XgbModel1.pickle','rb') as reader_xgb:
        xgb_model = pickle.load(reader_xgb)
    with open('SgdModel1.pickle','rb') as reader_sgd:
        sgd_model = pickle.load(reader_sgd)
    with open('LgbModel1.pickle','rb') as reader_lgb:
        lgb_model = pickle.load(reader_lgb)

    print("开始模型融合...")
    total_model = StackingModel(mod=(xgb_model,sgd_model,lgb_model,rf_model),meta_model = lr_model)
    print("模型融合初始化结束...")
    cv = check_cv(CV, y, classifier=True)
    scores = []
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
