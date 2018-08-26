

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.pipeline import make_pipeline
from model_tree import XGBClassifyCV,RandomForestClassifyCV,AveragingModels
from feature_handle import FeatureCreate,DataHandle,FeatureExtraction,FeatureMining
from utils import load_data,xgb_params,fit_xgb_params,CV,rf_params,rmse_cv,FeatureXgbHandle,PATH

def main():
    X_train, X_test, y_train, X_Id = load_data()

    FeatureXgbHandle(X_train,y_train)                                                   # XGB提取
    pipe = make_pipeline(
                (DataHandle()),                                                         # 数据处理
                (FeatureExtraction()),                                                  # 特征提取
                (FeatureMining()),                                                      # 特征挖掘
                (FeatureCreate())                                                       # 特征构造
             )

    '''
    拟合管道数据
    '''
    X_train = pipe.fit_transform(X_train,y_train)
    X_test  = pipe.fit_transform(X_test)

    '''
    初始化模型，由于我设计的LGB模型效果不好,我直接调用官方的LGB处理。
    '''
    xgb_model = XGBClassifyCV(xgb_params = xgb_params,fit_xgb_params = fit_xgb_params,cv = CV)
    rf_model  = RandomForestClassifyCV(rf_params=rf_params,cv = CV)
    lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=78,
                              learning_rate=0.01, n_estimators=1000, max_depth=13,
                              min_child_weight = 5, colsample_bytree = 0.9)
    score = rmse_cv(lgb_model, X_train, y_train)

    xgb_model.fit(X_train,y_train)
    rf_model.fit(X_train,y_train)
    lgb_model.fit(X_train,y_train)
    
    print("XGB 的 rmse集合为 :{}".format(xgb_model.cv_scores_))
    print("XGB 的 最佳rmse为 :{}".format(xgb_model.cv_score_))

    print("随机森林 的 rmse集合为 :{}".format(rf_model.cv_scores_))
    print("随机森林 的 最佳rmse为 :{}".format(rf_model.cv_score_))

    print("LBG 的 最佳rmse: {}".format(score.mean()))

    '''
    构建模型融合
    '''
    model_total = AveragingModels(models=(xgb_model,rf_model,lgb_model),weights = (0.525,0.075,0.35))
    model_total.fit(X_train, y_train)
    pre_end = np.expm1(model_total.predict(X_test))
    
    '''
    保存结果
    '''
    submission = pd.DataFrame()
    submission['ID'] = X_Id
    submission['target'] = pre_end
    submission.to_csv('submission_model_total_fifth_2018.csv', index=False)

    '''
    最后用你之前的方法，以及你的队友的结果，根据你们的得分，来调权重，或者通过Lasso算法拟合你们的Pre调权重。最后融合一起，防止过拟合的产生。
    '''
    s1 = pd.read_csv(PATH + 'submission_model_total_first_2018.csv')
    s2 = pd.read_csv(PATH + 'submission_model_total_second_2018.csv')
    s3 = pd.read_csv(PATH + 'submission_model_total_third_2018.csv')
    s4 = pd.read_csv(PATH + 'submission_model_total_fourth_2018.csv')
    s5 = pd.read_csv(PATH + 'submission_model_total_fifth_2018.csv')

    p1 = s1['target']
    p2 = s2['target']
    p3 = s3['target']
    p4 = s4['target']
    p5 = s5['target']

    p6 = 0.725 * p1 + 0.015 * p2 + 0.125 * p3 + 0.075 * p4 + 0.07 * p5

    sub = pd.DataFrame()
    sub['ID'] = s1['ID']
    sub['target'] = p6
    sub.to_csv('submission_model_total_end_2018.csv', index=False)

if __name__ == '__main__':
    main()
