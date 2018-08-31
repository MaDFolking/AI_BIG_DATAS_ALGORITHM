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

'''
panddas底层原理:
pandas存储dataframe中的真实数据，这些数据块都经过了优化。有个BlockManager类.
会用于保持行列索引与真实数据块的映射关系。他扮演一个API，提供对底层数据的访问。每当我们查询、编辑或删除数据时，
dataframe类会利用BlockManager类接口将我们的请求转换为函数和方法的调用。
每种数据类型在pandas.core.internals模块中都有一个特定的类。pandas使用ObjectBlock类来表示包含字符串列的数据块，
用FloatBlock类来表示包含浮点型列的数据块。对于包含数值型数据（比如整型和浮点型）的数据块，pandas会合并这些列，
并把它们存储为一个Numpy数组（ndarray）。Numpy数组是在C数组的基础上创建的，其值在内存中是连续存储的。基于这种存储机制，对其切片的访问是相当快的。
'''

'''
内存查看
'''
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep = True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep = True)
    usage_mb = usage_b / 1024 ** 2
    return "{:.2f} MB".format(usage_mb)

'''
降低pandas内存机制的函数
'''
def load_data():
    test = pd.read_csv('test.csv')
    list_tim = ['C1','C21']
    test.to_csv('feature_time.csv',columns = list_tim ,index=False)
    print(test.shape)
    '''
    查看内存分布
    '''
    print(test.info(memory_usage='deep'))
    '''
    pandas中各个数据类型平均内存使用量
    '''
    for i in ['float', 'int', 'object']:
        select_type = test.select_dtypes(include=[i])
        memory_mean = select_type.memory_usage(deep=True).mean() / 1024 ** 2
        print("{}类型的平均内存为: {:.3f} MB".format(i, memory_mean))

    '''
    查看不同类型数据范围
    '''
    int_types = ['uint8', 'int8', 'int16']
    for i in int_types:
        print("{}内存信息为{}".format(i, np.iinfo(i)))

    '''
    int类型优化数列。
    '''
    test_int = test.select_dtypes(include=['int'])
    test_down_int = test_int.astype('int8')

    print("原始int类型为: {},优化后int类型为: {}".format(mem_usage(test_int), mem_usage(test_down_int)))

    '''
    int类型优化可视化
    '''
    compare_ints = pd.concat([test_int.dtypes, test_down_int.dtypes], axis=1)
    compare_ints.columns = ['before', 'after']
    print("int类型表:{}".format(compare_ints.apply(pd.Series.value_counts)))


    '''
    float类型优化
    '''
    test_float = test.select_dtypes(include=['float64'])
    test_down_float = test_float.astype('float16')

    print("原始float类型为: {},优化后float类型为: {}".format(mem_usage(test_float), mem_usage(test_down_float)))

    '''
    float类型优化可视化
    '''
    compare_ints = pd.concat([test_float.dtypes, test_down_float.dtypes], axis=1)
    compare_ints.columns = ['before', 'after']
    print("float类型表:{}".format(compare_ints.apply(pd.Series.value_counts)))

    optimized_gl = test.copy()
    optimized_gl[test_down_int.columns] = test_down_int
    optimized_gl[test_down_float.columns] = test_down_float

    print("原始内存为: {},克隆的内存为: {}".format(mem_usage(test), mem_usage(optimized_gl)))

    gl_obj = test.select_dtypes(include=['object']).copy()
    print("object类型详细信息:", gl_obj.describe())

    '''
    obj类型优化
    '''
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:, col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:, col] = gl_obj[col]

    print("我们选取唯一值低于50的比较弱的特征作为修改为类别型")
    print("原始obj类型内存: {} , 优化后为: {}".format(mem_usage(gl_obj), mem_usage(converted_obj)))

    '''
    object类型优化可视化
    '''
    compare_obj = pd.concat([gl_obj.dtypes, converted_obj.dtypes], axis=1)
    compare_obj.columns = ['before', 'after']
    print("object内存表:", compare_obj.apply(pd.Series.value_counts))

    optimized_gl[converted_obj.columns] = converted_obj
    print(optimized_gl.dtypes)
    print("当前内存:", mem_usage(optimized_gl))

    dtypes = optimized_gl.dtypes
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]

    column_types = dict(zip(dtypes_col, dtypes_type))
    previes = first2pairs = {key: value for key, value in list(column_types.items())[:10]}
    import pprint
    pp = pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(previes)

    test = pd.read_csv('test.csv', dtype=column_types)
    print("现在内存为", mem_usage(test))
    print(test.shape)
    print(test.head())

    print(test.shape)
    train = pd.read_csv('train.csv', usecols = ['click'],nrows = 4577464 ,dtype = column_types)
    print(train.shape)
    print(train.head())
    test_new = pd.concat([test,train],axis = 1)
    print(test_new.describe())
    print(test_new.head())
    print(test_new.shape)
    test_new.drop('click',axis = 1,inplace = True)
    train_x,test_x,train_y,test_y = train_test_split(test_new,train,test_size = 0.2)
    print(train_x.shape," ",test_x.shape," ",train_y.shape," ",test_y.shape)
    return train_x,test_x,train_y,test_y
    
'''
拆分数据集,文件迭代法
'''
def load_data():
    train = pd.read_csv(PATH1+'train.csv', usecols=['hour'])
    print(train.shape)
    print(train.head())
    print(train.dtypes)
    list_train = []
    train.drop_duplicates(inplace=True)
    print(train.shape)
    print(train.index)
    for i in range(21,31):
        for j in train.index:
            # print(j)
            # print(train.iloc[j])
            j_new = train.loc[j].values
            print(j_new)
            # print(type(j_new))
            # print(j_new.shape)
            s = str(j_new)[5:7]
            # print(s)
            if int(s) > i:
                break
            j_new = int(j_new)
            print(j_new)
            if (14100000+i*100) < j_new:
                train.drop(j,axis = 0,inplace = True)
            else:
                continue

    print(train.shape)
    for i in train.index:
        list_train.append(i)

    for j in list_train:
        print(j)
    train1 = pd.read_csv(PATH1+'train.csv',iterator = True,index_col=False,engine = 'python')
    while True:
        try:
            for i in list_train:
                chunk = train1.get_chunk(i)
                chunk.to_csv(PATH2+'train{}.csv'.format(i), index=False, sep=',')
        except StopIteration:
            print("Iteration is stopped.")
            

'''
加载训练集
'''
def load_train_data():
    train_x = pd.read_csv(PATH2+'train4122995.csv')
    print(train_x.shape)
    print(train_x.columns)
    print(train_x[['C1','C20']])
    train_y = train_x.click
    print(train_y)
    train_x.drop('click',axis = 1,inplace = True)
    print(train_x.shape," ",train_y.shape)
    return train_x,train_y
    
'''
拆分数值型和非数值型函数
'''
def parse_numeric(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
    features = train_x.columns
    numeric_feature = train_x.columns[train_x.dtypes!='object']
    numeric_non_feature = [i for i in features if i not in numeric_feature]
    return features,numeric_feature,numeric_non_feature

'''
由于内存原因，我就不One-hot+PCA+随机森林分类了，具体可以看我的kaggle比赛项目和销量预测项目如何使用这个步骤。
'''

'''
非数值型编码
'''
def numeric_non_scale(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
    print("非数值型编码处理开始...")
    print(train_x.dtypes)
    #先LabelEncoder处理
    _,_,numeric_non_feature = parse_numeric(train_x)
    print(numeric_non_feature)
    le = LabelEncoder()
    for i in numeric_non_feature:
        le.fit(list(train_x[i]))
        train_x[i] = le.transform(train_x[i])
    print(train_x.dtypes)
    # one_hot = OneHotEncoder()
    # train_x = one_hot.fit_transform(train_x)
    print(train_x.head())
    print(train_x.shape)
    print(train_x.isnull().sum().sort_values(ascending=False))
    print("非数值型编码处理结束...")
    return train_x

'''
数值类型标准化
'''
def numeric_scale(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
    print("数值标准化处理开始...")
    _,numeric_feature,_ = parse_numeric(train_x)
    numeric_mean = train_x.loc[:,numeric_feature].mean()
    numeric_std = train_x.loc[:,numeric_feature].std()
    train_x.loc[:,numeric_feature] = (train_x.loc[:,numeric_feature]-numeric_mean)/numeric_std
    print(train_x.head())
    print(train_x.shape)
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    print(type(train_x.isnull().sum().sort_values(ascending=False).head()))
    print(type(train_x.isnull().sum().sort_values(ascending=False)[0]))
    #我们这次并没有nan值。所以函数里只是注释一些内容
    if train_x.isnull().sum().sort_values(ascending=False)[0]>0:
        parse_nan(train_x)
    print(train_x.head())
    print(train_x.shape)
    print(train_x.dtypes)
    feature, numeric_feature, _ = parse_numeric(train_x)
    if len(feature)==len(numeric_feature):
        print("数值标准化处理结束...")
        return train_x
    else:
        raise NotImplementedError("数值标准化处理失败...")

'''
去除nan值
'''
def parse_nan(train_x):
    if train_x is None:
        raise ValueError("输入值不能为空!")
    print("异常值处理开始...")
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    # train_x.CompetitionDistance.fillna(train_x.CompetitionDistance.mean(), inplace=True)
    # train_x.Open.fillna(1, inplace=True)
    train_x.fillna(0, inplace=True)
    print(train_x.isnull().sum().sort_values(ascending=False).head())
    print("异常值处理结束...")
