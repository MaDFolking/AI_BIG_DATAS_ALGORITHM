'''
特征挖掘
该项目中，时间只是涉及到该月22日-31日，所以我们分离出当天，当天的前一天，当天的后一天三类时间取比较。
再分，就是分上午，中午，下午三段时间，但是这样始终不是够精确，所以我们细致分成当前小时的前一小时，以及当前小时的后一小时。
后期如果维度太大，可以用聚类/PCA降维/随机森林分类法。
另外，我们上面数据探索中，可知道一些强类别点击率特别大，我们利用这些强类别，做一些特征性的组合，再通过平稳特征提取查看这些特征的强弱，看
这样组合是否有利于我们结果。最后可以用一个关键的强特，与其他特征用XGB做比较，查看rmse值，然后再进而做第二波特征组合，第二波特征组合主要给树型结果XGB,LGB使用。
而第一波主要给随机森林和FM使用。最后用LR来blending最后一步操作。
site:网站。 网站选取点击率高的作为强特征，然后，再与手机端，手机设备拼接看是否效果好。
app:手机端。 选取点击率高的的作为强特征，然后，再与网站，手机设备拼接看是否效果好。
device:手机。 手机类型，手机ID 选取高的作为强特征分离，然后，再与手机端，手机设备拼接看是否效果好。
'''
def mining_feature(train_x,train_y):
    assert len(train_x)>0
    train_y.to_csv('target.csv',index = False)
    for i in train_x.columns:
        train_x[i] = train_x[i].astype('float16')
    print(train_x.info(memory_usage = 'deep'))
    print("开始维度:",train_x.shape)
    print(train_x.dtypes)

    '''
    下面我们将主要特征都进行拼接，然后初始化为0，之后与每组最强的类别与这些拼接后的特征赋值为1.
    '''

    list_main = ['site_or_app','app_or_device','device_or_site']

    train_x['site_or_app'] = 0    # app或者网站
    train_x['app_or_device'] = 0  # app或者手机
    train_x['device_or_site'] = 0 # 手机或者网站
    for i in list_main:
        train_x[i] = train_x[i].astype('int8')
    site_list = train_x.site_id.apply(lambda x:x == '85f751fd').index
    app_list = train_x.app_id.apply(lambda x:x == 'ecad2386').index
    device_list = train_x.device_id.apply(lambda x:x == 'a99f214a').index

    #根据上面数据探索，我们将频率最高的一些类别特征性大的做一下强特征性处理。处理到我们新增加的维度当中，注意是:类别型数据。
    train_x.loc[site_list,'site_or_app'] = 1
    train_x.loc[site_list,'device_or_site'] = 1
    train_x.loc[app_list,'app_or_device'] = 1
    train_x.loc[app_list, 'site_or_app'] = 1
    train_x.loc[device_list,'device_or_site'] = 1
    train_x.loc[train_x.device_id.apply(lambda x:x == 'a99f214a').index, 'app_or_device'] = 1
    print("主要特征拼接后维度:", train_x.shape)

    '''
    下面是根据业务特征拼接
    我们将网站和app的id,区域类型做了拼接，看看是否为强特征。
    又考虑到手机自身model与ip,id可能也有关联，因为手机类型与不同用户，也可能会起到变化，虽然可能性不高，算是做一个
    挖掘尝试。
    最后尝试手机型号与app和网站id的拼接。看是否为强特征
    device_conn_type
    device_type
    '''
    list_business = ['site_app_id','site_app_domain','site_app_category','device_app_id','device_ip_model','device_model_id','device_site_id','device_type_app',
                     'device_conn_app','device_type_site','device_conn_site']
    train_x['site_app_id'] = np.add(train_x.site_id.values, train_x.app_id.values)
    train_x['site_app_domain'] = np.add(train_x.site_domain.values, train_x.app_domain.values)
    train_x['site_app_category'] = np.add(train_x.site_category.values, train_x.app_category.values)
    train_x['device_app_id'] = np.add(train_x.device_id.values, train_x.app_id.values)
    train_x['device_ip_model'] = np.add(train_x.device_model.values, train_x.device_ip.values)
    train_x['device_model_id'] = np.add(train_x.device_id.values, train_x.device_model.values)
    train_x['device_site_id'] = np.add(train_x.site_id.values, train_x.device_id.values)
    train_x['device_type_app'] = np.add(train_x.app_id.values, train_x.device_type.values)
    train_x['device_conn_app'] = np.add(train_x.app_id.values, train_x.device_conn_type.values)
    train_x['device_type_site'] = np.add(train_x.site_id.values, train_x.device_type.values)
    train_x['device_conn_site'] = np.add(train_x.site_id.values, train_x.device_conn_type.values)

    print("业务特征拼接后维度:", train_x.shape)
    '''
    下面是隐藏数据拼接进行挖掘。数据探索阶段，我们可以知道，C类型的除了C1,C18都是特征性比较强的，hour又是个强特征，所以除了C1都与hour进行拼接
    '''
    list_hide = ['C15_hour','C16_hour','C21_hour','C19_hour','C17_hour','C14_hour','C20_hour']

    train_x['C15_hour'] = np.add(train_x.C15.values,train_x.hour.values)
    train_x['C16_hour'] = np.add(train_x.C16.values, train_x.hour.values)
    train_x['C21_hour'] = np.add(train_x.C21.values, train_x.hour.values)
    train_x['C19_hour'] = np.add(train_x.C19.values, train_x.hour.values)
    train_x['C17_hour'] = np.add(train_x.C17.values, train_x.hour.values)
    train_x['C14_hour'] = np.add(train_x.C14.values, train_x.hour.values)
    train_x['C20_hour'] = np.add(train_x.C20.values, train_x.hour.values)


    print("隐藏特征拼接后维度:", train_x.shape)

    '''
    下面这个代码最后再加。
    '''
    #train_x = train_x.loc[np.logical_and(train_x.day.values>=21,train_x.day.values<32),:]
    '''
    做完后，这些就当我们的原始特征数，我们会再进行One-Hot+PCA处理，接下来，就开始挖掘树形算法的特征和线性算法的特征，用于最
    后的模型融合。
    '''


    for i in list_business+list_hide+list_main:
        train_x[i] = train_x[i].astype('float16')

    print(train_x.head())
    print(train_x.shape)
    print(train_x.info(memory_usage='deep'))

    '''
    保存特征数据
    '''
    train_x.to_csv('feature_001.csv',index=False)

    '''
    最后根据我们的特征性强弱，来用时间的前后天，小时来拼接强特征。这样方便我们后面做时间判断上的处理。这些特征作为必要特征，单独存放。
    可能跟时间挂钩的特征,因为ip地址相当于我们客户，所以通过ip,id来与时间想拼接。我们通过不同用户id，ip与时间拼接作为强特征。
    '''
    print("离散时间处理开始...")
    nn1 = train_x.shape[1]
    train_x['day'] = np.round(train_x.hour % 10000 / 100)
    print(train_x['day'].head())
    train_x['hour1'] = np.round(train_x.hour % 100)
    print(train_x['hour1'].head())
    train_x['day_hour'] = (train_x.day.values - 21) * 24 + train_x.hour1.values
    print(train_x['day_hour'].head())
    train_x['day_hour_prev'] = train_x['day_hour'] - 1
    print(train_x['day_hour_prev'].head())
    train_x['day_hour_next'] = train_x['day_hour'] + 1
    print(train_x['day_hour_next'].head())
    train_x['device_ip_day_device_id'] = np.add(train_x.device_ip.values,train_x.day.values,train_x.device_id.values)
    train_x['device_ip_day_app_id'] = np.add(train_x.device_ip.values, train_x.day.values, train_x.device_id.values)
    train_x['device_ip_day_site_id'] = np.add(train_x.device_ip.values, train_x.day.values, train_x.device_id.values)
    '''
    然后是前一个小时的拼接
    '''
    train_x['device_ip_pre_hour_device_id'] = np.add(train_x.device_ip.values,train_x.day_hour_prev.values,train_x.device_id.values)
    train_x['device_ip_pre_hour_app_id'] = np.add(train_x.device_ip.values, train_x.day_hour_prev.values, train_x.device_id.values)
    train_x['device_ip_pre_hour_site_id'] = np.add(train_x.device_ip.values, train_x.day_hour_prev.values, train_x.device_id.values)
    '''
    后一个的拼接
    '''
    train_x['device_ip_next_hour_device_id'] = np.add(train_x.device_ip.values,train_x.day_hour_next.values,train_x.device_id.values)
    train_x['device_ip_next_hour_app_id'] = np.add(train_x.device_ip.values, train_x.day_hour_next.values, train_x.device_id.values)
    train_x['device_ip_next_hour_site_id'] = np.add(train_x.device_ip.values, train_x.day_hour_next.values, train_x.device_id.values)
    '''
    进行相减
    这里是标准化后的数据，让线性关系扩大，再减去平稳线性关系1.使其更平滑
    '''
    train_x['diff_hour_device_ip_next_hour_device_id'] = (train_x.device_ip_next_hour_device_id.values - train_x.device_ip_pre_hour_device_id.values)*((train_x.device_model_id*2-1))
    train_x['diff_hour_device_ip_next_hour_app_id'] = (train_x.device_ip_next_hour_device_id.values - train_x.device_ip_pre_hour_app_id.values)*((train_x.device_app_id*2-1))
    train_x['diff_hour_device_ip_next_hour_site_id'] = (train_x.device_ip_next_hour_device_id.values - train_x.device_ip_pre_hour_site_id.values)*((train_x.device_site_id*2-1))
    list_time = ['device_ip_day_device_id','device_ip_day_device_id','device_ip_day_device_id',
                'device_ip_pre_hour_device_id','device_ip_pre_hour_app_id','device_ip_pre_hour_site_id',
                'device_ip_next_hour_device_id','device_ip_next_hour_app_id','device_ip_next_hour_site_id',
                'diff_hour_device_ip_next_hour_device_id','diff_hour_device_ip_next_hour_app_id','diff_hour_device_ip_next_hour_site_id']

    for i in list_time:
        train_x[i] = train_x[i].astype('float16')
    print("离散时间后维度:", train_x.shape)
    print("开始保存时间业务数据...")
    train_x.to_csv('feature_time.csv',columns = list_time ,index=False)
    return train_x
    
'''
特征离散，取点击率大于30000的类别离散，如果机器内存足够大，可以选取更少的点击率
'''
def parse_hot(train_x):
    assert len(train_x)>0
    print("类别离散化处理开始...")
    _, _, numeric_non_feature = parse_numeric(train_x)
    print(train_x.shape)
    print(len(numeric_non_feature))
    # 再 One - Hot 处 理 , 只处理频率大于20000的进行离散化类别。
    for i in numeric_non_feature:
        feature_num = train_x[i].value_counts()[lambda x:x>30000].index
        print(len(feature_num))
        all_num = pd.get_dummies(feature_num)
        train_x = pd.concat((train_x,all_num),axis=1)
        print(train_x.shape)
    print(train_x.shape)
    if train_x.shape[1]<50:
        raise NotImplementedError("类别离散化处理失败")
    else:
        print("类别离散化处理结束...")
        return train_x
        
'''
特征选择
删除一列只有一个数的特征
删除俩个列完全相同的特征
VarianceThreshold删除所有方差过低的列,因为方差太低特征性太差了。
'''
def feature_selection(train_x):
    assert len(train_x)>0
    print("特征选择处理开始...")
    #检查异常点nan或inf
    print(np.all(np.isfinite(train_x)))
    train_x.drop('id',axis = 1,inplace = True)
    parse_nan(train_x)
    selector = VarianceThreshold(0.01)
    selector.fit_transform(train_x)
    print(train_x.shape)
    # 删除唯一性数据，这种数据没意义，特征性为0
    uniques = train_x.columns[train_x.nunique()==1]
    if len(uniques)>0:
        train_x.drop(uniques, axis=1, inplace=True)

    # 删除俩列相同的数据
    all_columns = train_x.columns
    colsToRemove = []
    for i in range(len(all_columns) - 1):
        m = train_x[all_columns[i]].values
        for j in range(i + 1, len(all_columns)):
            if np.array_equal(m, train_x[all_columns[j]].values):
                colsToRemove.append(all_columns[j])

    train_x.drop(colsToRemove, axis=1, inplace=True)
    print(train_x.shape)
    print("特征选择处理结束...")
    return train_x
    
'''
特征拼接
'''
def feature_union(train_x):
    assert len(train_x)>0
    print(train_x.dtypes)
    train_x.astype('float16')
    print(train_x.dtypes)
    '''
    拼接用户与广告类别的特征，这个是用作 state 的状态改变而拼接。
    '''
    train_x['C1_site_id'] = np.add(train_x.C1.values,train_x.site_id.values).astype('float16')
    train_x['C1_app_id'] = np.add(train_x.C1.values, train_x.app_id.values)
    train_x['C1_device_conn_type'] = np.add(train_x.C1.values, train_x.device_conn_type.values)

    train_x['C14_site_id'] = np.add(train_x.C14.values, train_x.site_id.values)
    train_x['C14_app_id'] = np.add(train_x.C14.values, train_x.app_id.values)
    train_x['C14_device_conn_type'] = np.add(train_x.C14.values, train_x.device_conn_type.values)

    train_x['C17_site_id'] = np.add(train_x.C17.values, train_x.site_id.values)
    train_x['C17_app_id'] = np.add(train_x.C17.values, train_x.app_id.values)
    train_x['C17_device_conn_type'] = np.add(train_x.C17.values, train_x.device_conn_type.values)

    train_x['C19_site_id'] = np.add(train_x.C19.values, train_x.site_id.values)
    train_x['C19_app_id'] = np.add(train_x.C19.values, train_x.app_id.values)
    train_x['C19_device_conn_type'] = np.add(train_x.C19.values, train_x.device_conn_type.values)

    train_x['C20_site_id'] = np.add(train_x.C20.values, train_x.site_id.values)
    train_x['C20_app_id'] = np.add(train_x.C20.values, train_x.app_id.values)
    train_x['C20_device_conn_type'] = np.add(train_x.C20.values, train_x.device_conn_type.values)

    train_x['C21_site_id'] = np.add(train_x.C21.values, train_x.site_id.values)
    train_x['C21_app_id'] = np.add(train_x.C21.values, train_x.app_id.values)
    train_x['C21_device_conn_type'] = np.add(train_x.C21.values, train_x.device_conn_type.values)

    '''
    离散时间
    '''
    train_x['day'] = np.round(train_x.hour % 10000 / 100)
    train_x['hour1'] = np.round(train_x.hour % 100)
    train_x['day_hour'] = (train_x.day.values - 21) * 24 + train_x.hour1.values
    # train_x['day_hour_prev'] = train_x['day_hour'] - 1
    # train_x['day_hour_next'] = train_x['day_hour'] + 1
    train_x.drop('day',axis = 1,inplace = True)
    train_x.drop('hour1', axis = 1,inplace = True)

    '''
    拼接不同时间段的不同类别广告点击率，当做action使用。这步在强化学习时再做。
    '''
    train_x['C1_hour'] = np.add(train_x.C1.values, train_x.hour.values).astype('float16')
    train_x['C14_hour'] = np.add(train_x.C14.values, train_x.hour.values)
    train_x['C17_hour'] = np.add(train_x.C17.values, train_x.hour.values)
    train_x['C19_hour'] = np.add(train_x.C19.values, train_x.hour.values)
    train_x['C20_hour'] = np.add(train_x.C20.values, train_x.hour.values)
    train_x['C21_hour'] = np.add(train_x.C21.values, train_x.hour.values)

    print(train_x.shape)
    return train_x
    
 '''
 线性特征提取
 '''
 def feature_extraction_RandomLasso(flag = True):
    from sklearn.linear_model import RandomizedLasso
    if flag == True:
        X_train = pd.read_csv('feature_001.csv')
        X_train.drop('id',axis = 1,inplace = True)
        X_train = parse_nan(X_train)
        y_train = pd.read_csv('target.csv')
        print(type(X_train))
        for i in X_train.columns:
            X_train[i] = X_train[i].astype('float16')
        print(X_train.info(memory_usage = 'deep'))
        print(y_train.info(memory_usage = 'deep'))
        print("稳定性选择法提取特征开始...")
        #print(X_train.isnull().sum().sort_values(ascending=False).head())
        NUM = 20
        randomLasso = RandomizedLasso()
        randomLasso.fit(X_train, y_train)
        features = randomLasso.scores_
        score = X_train.columns
        print(features)
        print(sorted(zip(map(lambda x:round(x,4),features),score),reverse = True))
        featureList = sorted(zip(map(lambda x:round(x,4),features),score),reverse = True)
        featureList = [i[1] for i in featureList][:NUM]
        X_train = X_train[featureList]
        print(X_train.shape)
        if X_train.shape[1]!= NUM:
            raise NotImplementedError("稳定性选择法提取特征处理失败")
        print("稳定性选择法提取特征结束...")
        X_train.to_csv('feature_tree_end.csv')
    else:
        X_train = pd.read_csv('feature_linear_end.csv')
        y_train = pd.read_csv('target.csv')
        X_train.drop('id',axis = 1,inplace = True)
        X_train = parse_nan(X_train)
        print("稳定性选择法提取特征开始...")
        print(X_train.isnull().sum().sort_values(ascending=False).head())
        NUM = 30
        randomLasso = RandomizedLasso()
        randomLasso.fit(X_train, y_train)
        features = randomLasso.scores_
        score = X_train.columns
        print(features)
        print(sorted(zip(map(lambda x:round(x,4),features),score),reverse = True))
        featureList = sorted(zip(map(lambda x:round(x,4),features),score),reverse = True)
        featureList = [i[1] for i in featureList][:NUM]
        X_train = X_train[featureList]
        print(X_train.shape)
        if X_train.shape[1]!= NUM:
            raise NotImplementedError("稳定性选择法提取特征处理失败")
        print("稳定性选择法提取特征结束...")
        X_train.to_csv('feature_linear_best.csv')
    return X_train

    
 '''
XGB比较树形特征,这部分特征进行进一步构造，提取，然后给树型结构LGB和XGB使用。
https://blog.csdn.net/a358463121/article/details/77993198?locationNum=10&fps=1
'''
from xgboost import XGBRegressor
from xgboost import XGBClassifier
def xgb_feature():
    train_x = pd.read_csv('feature_001.csv')
    train_y = pd.read_csv('target.csv')
    train_x.drop('id',axis = 1,inplace = True)
    for i in train_x.columns:
        train_x[i] = train_x[i].astype('float16')
    print(train_x.shape)
    print(train_x.info(memory_usage='deep'))
    if train_x is None or train_y is None:
        raise ValueError('输入值不能为空!')
    print("特征提取开始...")

    xgb_model = XGBClassifier(n_estimators=100, n_jobs=-1, nthread=-1)
    folds = KFold(4, True, 134259)
    kf = [(trn_, val_) for trn_, val_ in folds.split(train_x)]
    scores = []
    '''
    day	SchoolHoliday	Promo2	Promo	DayOfWeek	CompetitionOpenSinceMonth	CompetitionDistance	Assortment
    Promo2SinceWeek	Promo2SinceYear	StoreType	Store	aug	dec	CompetitionOpenSinceYear
    '''
    X_train_tmp = train_x
    features_s = X_train_tmp.drop('device_conn_type',axis = 1).columns.values
    print(features_s)
    '''
    我们选取一个强特征，跟下面循环每个特征拼接然后做拟合。
    '''
    X_train = train_x
    print(X_train.dtypes)
    print(X_train.shape)
    print(X_train.shape)
    print(X_train.isnull().sum().sort_values(ascending=False).head())


    # feature_new = X_train_tmp.columns.values
    # print(feature_new)
    '''

    '''

    for _f in features_s:
        score = 0
        for trn_, val_ in kf:
            print(_f)
            print(len(trn_))  #训练集每次随机4份 总共5份
            print(len(val_))  #测试集每次随机1份 总共5份
            print( X_train[['device_conn_type',_f]].iloc[trn_].shape)
            print(train_y.iloc[trn_].shape)
            xgb_model.fit(
                X_train[['device_conn_type',_f]].iloc[trn_], train_y.iloc[trn_],
                eval_set=[(X_train[['device_conn_type',_f]].iloc[val_], train_y.iloc[val_])],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=False
            )
            '''
            ntree_limit:在预测中限制树的数量；默认为0（使用所有树）。
            best_ntree_limit:大概意思就是获取正确树的值。
            因为n_splits为迭代次数，所以最后score取总和除以它就是平均数
            '''
            print(folds.n_splits)  #我们总共有5份，n_splits就是4，也就是迭代次数 而splits就是这5份中训练集和测试的坐标值。
            predict = xgb_model.predict(X_train[['device_conn_type',_f]].iloc[val_],ntree_limit=xgb_model.best_ntree_limit)
            score = score + metrics.roc_auc_score(train_y.iloc[val_], predict)
        score /= folds.n_splits
        scores.append((_f,score))
    '''
    保存特征前，别忘了设置列名columns=['featureImportance', 'rmse']
    '''
    featureImportance = pd.DataFrame(scores, columns=['featureImportance', 'auc']).set_index('featureImportance')
    featureImportance.sort_values(by='auc', ascending=True, inplace=True)

    featureImportance.to_csv('featureImportance_xgb.csv', index=True)

    print("特征提取结束...")
    print("*"*50)
    return X_train



'''
线性模型构建特征，这部分特征给FM,LR提取，然后进一步构造。

'''
def linear_build_feature(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
    '''
    由上面得知线性重要特征，和原始重要特征，开始构建线性重要特征。
    '''
    list_linear_feature = ['site_app_category', 'C15_hour','site_category',  'app_id', 'C17', 'C14', 'device_app_id']
    list_import_feature = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
    train_x['C17_C14'] = np.add(train_x.C17.values,train_x.C14.values)
    train_x['C17_device_type'] = np.add(train_x.C17.values, train_x.device_type.values)
    train_x['C17_site_category'] = np.add(train_x.C17.values, train_x.site_category.values)
    train_x['C15_hour_site_category'] = np.add(train_x.C15_hour.values, train_x.site_category.values)
    train_x['site_app_category_C15_hour'] = np.add(train_x.site_app_category.values, train_x.C15_hour.values)
    train_x['app_id_C17'] = np.add(train_x.app_id.values, train_x.C17.values)

    list_total_linear = ['C17_C14','C17_device_type','C17_site_category','C15_hour_site_category','site_app_category_C15_hour','app_id_C17']
    for i in list_total_linear:
        train_x[i] = train_x[i].astype('float16')

    print(train_x.shape)
    print(train_x.info(memory_usage = 'deep'))
    train_x.to_csv("feature_tree_best.csv", index=False)
    return train_x

'''
XGB与LGB模型构建特征，这部分特征给FM,LR提取，然后进一步构造。

'''
def tree_build_feature():

    train_x = pd.read_csv('feature_001.csv')
    '''
    由上面得知线性重要特征，和原始重要特征，开始构建树形重要特征。
    '''
    list_linear_feature = ['app_id', 'app_domain','app_category',  'C14_hour', 'device_id', 'device_model', 'device_type']
    list_import_feature = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_ip', 'device_conn_type']
    train_x['device_type_of_device_model'] = np.add(train_x.device_type.values,train_x.device_model.values)
    train_x['device_type_of_device_id'] = np.add(train_x.device_type.values, train_x.device_id.values)
    train_x['device_type_of_app_category'] = np.add(train_x.device_type.values, train_x.app_category.values)
    train_x['device_type_of_app_domain'] = np.add(train_x.device_type.values, train_x.app_domain.values)
    train_x['device_type_of_app_id'] = np.add(train_x.device_type.values, train_x.app_id.values)

    list_total_linear = ['device_type_of_device_model','device_type_of_device_id',
                         'device_type_of_app_category','device_type_of_app_domain','device_type_of_app_id']
    list_importance_feature = [
          'C20_hour',  'site_app_category',  'hour', 'C15_hour',
          'site_category', 'app_id',  'C17', 'C14',  'device_app_id', 'C21_hour', 'C20',  'C19_hour', 'banner_pos',  'C1', 'day_hour',
        'C15', 'C16','C18','C19','C21','device_ip','device_conn_type'
    ]
    for i in train_x.columns:
        train_x[i] = train_x[i].astype('float16')

    print(train_x.shape)
    print(train_x.info(memory_usage = 'deep'))
    train_x.to_csv("feature_xgb_lgb_end.csv",columns=list_importance_feature+list_total_linear, index=False)
    return train_x
    
'''
最后需要将特征整理,分别用树形和线性去拟合。
'''

