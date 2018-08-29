

import time
import random
import pandas as pd

from nltk.stem.porter import *
from nltk.metrics import edit_distance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer


start_time = time.time()
stemmer = PorterStemmer()
random.seed(2016)

PATH = ''                                                                   # 数据路径
SEED = 2018                                                                 # 随机种子
STOP_WORDS = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku',                 # 本次项目的停止词
              'with', 'what', 'from', 'that', 'less', 'er' ,'ing']
STR_NUM = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,            # 我用英文配上值的格式，将英文对应里的数字。
           'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}          # 要最终转化为字符串，在提取词向量前的操作

'''
util包下的函数
'''

'''
加载数据
'''
def load_data():
    df_train = pd.read_csv(PATH + 'train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv(PATH + 'test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv(PATH + 'product_descriptions.csv')
    df_attr = pd.read_csv(PATH + 'attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    num_train = df_train.shape[0]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    return df_all,num_train,df_brand,df_attr,df_pro_desc,df_test,df_train

'''
提取词性,记住一定要先判断是否为字符串。
下面是我按照通常英文的处理方式，处理后直接将数字转为字符串，然后stem提取词性。
随后，该项目中一些特殊的很难区分的英文词，做下处理。
'''
def str_stem(s):
    if isinstance(s ,str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)

        s = s.lower()
        s = s.replace("  ", " ")
        s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
        s = s.replace(",", " ")
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°", " degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")
        
        # 开始提取词性
        s = (" ").join([str(STR_NUM[z]) if z in STR_NUM else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
        # 根据数据中一些商品词汇与英文违背的词汇，做一些处理。
        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air condition")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        return s
    else:
        return "null"

'''
比较俩个文本相近的词，利用倒叙查找法，过滤掉英文的噪点词，去倒叙相同的词汇。
'''
def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

'''
将第一个字符串倒叙依次截取，跟第二个数组里的字符串一一比较，如果一样则放入到返回的列表中。
这是个常见的英文相似度表示法，类似 abvdasda = aba 这样的，我们是为了去除噪点而设计的函数。
'''
def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                s = s[len(s) - j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r


'''
取公有的词汇。
str1.split() 可以理解为生成一个列表，虽然字符串本质就是列表。
'''
def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
        # new for edit distance
        if cnt == 0 and len(word) > 3:
            s1 = [z for z in list(set(str2.split(" "))) if abs(len(z) - len(word)) < 2]
            t1 = sum([1 for z in s1 if edit_distance(z, word) < 2])
            if t1 > 1:
                cnt += 0.5
    return cnt

'''
俩个字符串是否相同。str1是否包含子字符串str2.如果包含，则返回开始索引值，这里是0，如果不相同，返回-1. 但不相同时，我们直接返回
cnt 了，最后我们统计相同的个数。然后i不停的加子字符串长度，直到等于总字符串退出循环。
'''
def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

'''
评估指标
'''
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

'''
删除原始特征，因为使用管道时，我们返回的是numpy.array。所以最后是.values
'''
class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'product_info', 'attr',
                       'brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches

'''
取我们想要传入的特征，返回的是DataFrame对象。
'''
class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)











