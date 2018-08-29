

import numpy as np
import pandas as pd

from utils import str_stem,seg_words,str_common_word,str_whole_word,load_data

'''
在NLP中，主要就是拼接和构造特征，所以这里我直接都写了。
'''

'''
特征工程(特征拼接与特征构造，主要是增加相似度的特征)
'''
def feature_transform():
    # 加载数据集
    df_all, num_train, df_brand, df_attr, df_pro_desc, df_test, df_train = load_data()
    # 提取客户搜索的词语词性
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
    # 提取产品标题的词语词性
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
    # 提取产品描述的词语词性
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_stem(x))
    # 按照业务需求，把brand改成value了。
    df_all['brand'] = df_all['brand'].map(lambda x: str_stem(x))
    
    '''
    特征拼接，搜索的内容+搜索后标题+产品描述进行拼接,这步是为我们之后特征构造和特征拼接准备。
    '''
    df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
    
    '''
    下面是特征构造部分。
    '''
    
    # (1) 取搜索的长度，因为我们推断，搜索词的长度可能跟结果有一定关联。x.split()是字符串直接分割成列表格式的技巧。
    df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
    # 同理标题长度。
    df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
    # 描述长度。
    df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
    # 产品属性长度
    df_all['len_of_brand'] = df_all['brand'].map(lambda x: len(x.split())).astype(np.int64)
    # 比较search_term 和 product_title的相似度，赋值给search_term
    df_all['search_term'] = df_all['product_info'].map(lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))
    # 物品标题是否有词在搜索词里出现，有则返回俩者相同词的个数。
    df_all['query_in_title'] = df_all['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
    # 物品标题和物品描述是否有相同的，有则返回俩者相同词的个数。
    df_all['query_in_description'] = df_all['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
    # 物品标题的最后一个单词是否跟搜索词里出现。 因为假如你搜电脑 可能出现 XXX的电脑。如果出现返回+1 即可。
    df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
    # 查看物品标题的最后一个单词是否跟产品描述里出现。 因为假如你搜电脑 可能出现 XXX的电脑。如果出现返回+1 即可。
    df_all['query_last_word_in_description'] = df_all['product_info'].map(
        lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
    # 取物品标题与搜索词里共同部分，返回个数。
    df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
    # 取物品标题用户物品描述共同部分，返回个数。
    df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
    # 标题搜索词共同部分长度 除以 搜索词自身长度。
    df_all['ratio_title'] = df_all['word_in_title'] / df_all['len_of_query']
    # 标题描述共同部分长度 除以 搜索词自身长度。
    df_all['ratio_description'] = df_all['word_in_description'] / df_all['len_of_query']
    # 物品属性拼接 搜索词+属性特征
    df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']
    # 物品属性与搜索词共同部分的长度。
    df_all['word_in_brand'] = df_all['attr'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
    # 同理共同长度除以原始长度
    df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['len_of_brand']
    # ravel  将展平的底层数据作为ndarray返回 这里是取属性的胃一直。也就是我们要取词向量个数了。
    df_brand = pd.unique(df_all.brand.ravel())
    d = {}    # 将词向量长度，也就是特征数赋值给我们的词向量，所以用字典表示。
    i = 1000  # 词向量长度，也就是特征数，one-hot处理后的稀疏矩阵。
    for s in df_brand:
        d[s] = i
        i += 3
    # 构造属性词向量特征，和我们要预测的搜索词词向量特征
    df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
    df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))
    # 拆分数据集。
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]

    y_train = df_train['relevance']
    id_test = df_test['id']
    # 将原始属性删除，以及没必要的属性，最后只剩下我们的相似性的数据以及文本特征数据。
    d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'product_info', 'attr',
                   'brand']
    train2 = df_train[d_col_drops]
    test2 = df_test[d_col_drops]
    train = df_train.drop(d_col_drops, axis=1)[:]
    test = df_test.drop(d_col_drops, axis=1)[:]
    return train,test,train2,test2,y_train,id_test

'''
编码处理
'''
def Hot_Handle():
    train, test, train2, test2, y_train, id_test = feature_transform()
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
        if train_series.dtype == 'O':
            train[train_name], tmp_indexer = pd.factorize(train[train_name])  # 将对象编码为枚举类型或分类变量。
            test[test_name] = tmp_indexer.get_indexer(test[test_name])
        else:
            tmp_len = len(train[train_series.isnull()])
            if tmp_len > 0:
                train.loc[train_series.isnull(), train_name] = train_series.mean()
            tmp_len = len(test[test_series.isnull()])
            if tmp_len > 0:
                test.loc[test_series.isnull(), test_name] = train_series.mean()

    train = pd.concat([train, train2], axis=1)[:]
    test = pd.concat([test, test2], axis=1)[:]
    return train,test,y_train,id_test












