

import pandas as pd
import xgboost as xgb

from feature_handle import Hot_Handle
from sklearn.pipeline import FeatureUnion
from sklearn import pipeline, model_selection
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import SEED,cust_regression_vals,cust_txt_col,RMSE

'''
这次我们用网格搜索来完成,所以我就不设计XGB类了。
'''


if __name__ == '__main__':
    # 直接调用特征工程
    train, test, y_train, id_test = Hot_Handle()
    xgb_model = xgb.XGBRegressor(learning_rate=0.25, booster='gbtree', silent=False, objective="reg:linear", nthread=-1,
                                 gamma=1.25,
                                 min_child_weight=1, max_delta_step=0,
                                 subsample=0.9, colsample_bytree=0.8, colsample_bylevel=1, reg_alpha=0.0001, reg_lambda=1.4,
                                 scale_pos_weight=1, base_score=0.5, seed=SEED, missing=None)

    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state=SEED)


    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            # transformer_list 要应用于数据的变换器对象的列表。每个元组的前半部分是变换器的名称。 下面四个管道开始处理四个特征，都用tf计算相似性，再用tsvd进行矩阵降维。
            transformer_list=[
                # 确保特征拼接同一。删除没必要的特征。
                ('cst', cust_regression_vals()), 
                ('txt1',
                 pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                (
                    'txt2',
                    pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                ('txt3',
                 pipeline.Pipeline(
                     [('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
            ],
            # 每个变压器的特征的乘法权重。键是变换器名称，值是权重。
            transformer_weights={
                'cst': 1.0,
                'txt1': 0.5,
                'txt2': 0.25,
                'txt3': 0.0,
                'txt4': 0.5
            },
            n_jobs=4
        )),
        ('xgb_model', xgb_model)])

    param_grid = {'xgb_model__max_depth': [5,7,9,11,13,15], 'xgb_model__n_estimators': [10,100,200,300,400,500,1000]}
    model = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=20,
                                         scoring=RMSE)
    model.fit(train, y_train.values)

    print("最好的参数为:")
    print(model.best_params_)
    print("最好的score为:")
    print(model.best_score_)

    y_pred = model.predict(test)


    min_y_pred = min(y_pred)
    max_y_pred = max(y_pred)
    min_y_train = min(y_train.values)
    max_y_train = max(y_train.values)
    print(min_y_pred, max_y_pred, min_y_train, max_y_train)

    # 根据业务描述，我们必须取1-3之间。
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0

