
#！/user/bin/env python3
# -*- coding:utf-8 -*-
import re
import os
import sys
import time
import array
import psutil
import logging
import unidecode
import numpy as np

from string import digits
from functools import wraps
from scipy import sparse as sp
from nltk import PorterStemmer
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _make_int_array, CountVectorizer

'''
公共变量
'''

'''
设定最小价格和最大价格
'''
def on_kaggle():
    return "kaggle/working" in os.getcwd()

PATH = ''                                                # 路径，需要自己设定。
stemmer = PorterStemmer()                                # 初始化词干
digits_set = set(digits)                                 # 将所有数字放在一起，去重处理，方便以后使用
_with_spaces = re.compile(r"\s\s+")                     # 匹配空格
regex_double_path = re.compile('[-]{2,}')                # 不加r时，就是split意思，这里间隔开超过2个-的格式。
UNK = 'unk'                                              # 方块的意思，这是我们填充缺省值用的变量
ITEM_DESCRIPTION_MAX_LENGTH = 1050                       # 文本向量的最大长度,也就是每个特征下标签最大内容,如果超过去就过滤掉。
NAME_MAX_LENGTH = 100                                    # 列表标题名的最大长度，超过这个长度当做异常处理
BRAND_NAME_MAX_LENGTH = 100                              # 品牌名字的最大长度，超过这个长度当做异常处理
TEST_CHUNK = 350000                                      # 读取测试集最多文件的iter,用来节约内存使用。 这个先待定，后续根据情况使用
MIN_PRICE = 3                                            # 价格最低值，低于这个值当做异常处理
MAX_PRICE = 10000                                        # 价格最大值，高于这个值当做异常处理
MIN_PRICE_PRED = 3                                       # 矩阵clip取范围操作时的最小价格值
MAX_PRICE_PRED = 2000                                    # 矩阵clip取范围操作时的最大价格值
MEAN_LOG_PRICE = 2.9806                                  # 商品loggle价值，这个先待定。

'''
下面都是设置系统环境变量: os.environ.get()
'''
N_CORES = 4 if on_kaggle() else 7
DEBUG_N = int(os.environ.get('DEBUG_N', 0))
TEST_SIZE = int(os.environ.get('TEST_SIZE', 1))
VALIDATION_SIZE = float(os.environ.get('VALIDATION_SIZE', 0.05))
DUMP_DATASET = int(os.environ.get('DUMP_DATASET', 0))
USE_CACHED_DATASET = int(os.environ.get('USE_CACHED_DATASET', 0))
HANDLE_TEST = int(os.environ.get('HANDLE_TEST', 1))
DEBUG = DEBUG_N > 0

'''
log 日志
'''
def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log-{os.getpid()}.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('mercari')



'''
公共函数
'''

'''
拼接排序后的单词
'''
def word_to_charset(word):
    return ''.join(sorted(list(set(word))))


'''
匹配年份，找到所有四位数的年份:re.findall('[0-9]{4}') 然后截取到1970-2018
最后我们要匹配最大的年份。也是过滤作用，防止数据里的异常点。
'''
def extract_year(text):
    text = str(text)
    matches = [int(year) for year in re.findall('[0-9]{4}', text)
               if int(year) >= 1970 and int(year) <= 2018]
    if matches:
        return max(matches)
    else:
        return 0

'''
判断文本中每个样本对象是否含有数字。any表示对象是为空。如果有数字，需要做一些处理，比如这个商品是3X,3XL。 这样它的价格将会受到波动。
'''
def has_digit(text):
    try:
        return any(c in digits_set for c in text)
    except:
        return False

'''
转化为float类型，转化失败就设置为0，因为后续我们有大量转化为float操作，需要设置一下，并设置异常。
'''
def try_float(t):
    try:
        return float(t)
    except:
        return 0

'''
打印时间函数
'''
class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        logger.info('Starting {}'.format(self.message))
        self.start_clock = time.clock()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_clock = time.clock()
        self.end_time = time.time()
        self.interval_clock = self.end_clock - self.start_clock
        self.interval_time = self.end_time - self.start_time
        logger.info('Finished {}. Took {:.2f} seconds, CPU time {:.2f}, effectiveness {:.2f}'.format(
            self.message, self.interval_time, self.interval_clock, self.interval_clock / self.interval_time))

'''
我们的基线，rmsle
'''
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

'''
log日志显示时间
'''
def log_time(fn, name):
    @wraps(fn)
    def deco(*args, **kwargs):
        logger.info('[{name}] << starting {fn.__name__}')
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.time() - t0
            logger.info('[{name}] >> finished {fn.__name__} in {dt:.2f} s, '
                         '{memory_info()}')
    return deco

'''
内存信息
'''
def memory_info():
    process = psutil.Process()
    memory_info = process.memory_info()
    return ('process {process.pid}: RSS {memory_info.rss:,} {memory_info}; '
            'system: {psutil.virtual_memory()}')

'''
二值化处理
'''
def binarize(ys, percentiles, soft=True, soft_scale=160):
    if soft:
        mean_percentiles = get_mean_percentiles(percentiles)
        binarized = np.exp(-0.5 * soft_scale * (
                (np.expand_dims(ys, 1) - mean_percentiles) ** 2))
        binarized = (binarized.T / binarized.sum(axis=1)).T
    else:
        binarized = np.zeros((len(ys), len(percentiles) - 1), dtype=np.float32)
        for i in range(1, len(percentiles)):
            binarized[:, i - 1] = (
                    (ys > percentiles[i - 1]) & (ys <= percentiles[i]))
    return binarized

'''
设置百分位数，用于后面数据集处理
np.percentile:百分位数。
 (list(np.range(0, 100, 100 / n_bins)) + [100])
'''
def get_percentiles(ys, n_bins):
    return np.percentile(ys, list(np.arange(0, 100, 100 / n_bins)) + [100])

'''
百分位数的平均数
'''
def get_mean_percentiles(percentiles):
    return np.mean([percentiles[:-1], percentiles[1:]], axis=0)



'''
提取词干
'''
def stem_word(word):
    return stemmer.stem(word)



'''
去掉文本中的异常符号
'''
def clean_text(text,tokenizer,hashchars = False):
    text = str(text).lower()
    text = _with_spaces.sub(" ",text)
    text = unidecode.unidecode(text)
    '''
    下面是处理 - 俩个- 还有:)这些，都是按照业务规定或者我们的模型来处理，基本都是被空格取代，或者simley,方便后序模型计算。
    '''
    text = text.replace(' -', ' ').replace('- ', ' ').replace(' - ', ' ')
    text = re.compile('[-]{2,}').sub(' ', text)
    text = text.replace(':)', 'smiley').replace('(:', 'smiley').replace(':-)', 'smiley')
    '''
    都变为小写，速度更快，内存更少。
    '''
    tokens = tokenizer(str(text).lower())
    if hashchars:
        tokens = [word_to_charset(t) for t in tokens]
    return "".join(map(stem_word, tokens)).lower()

'''
设置文本内容最大长度
'''
def trim_description(text):
    if text and isinstance(text, str):
        return text[:1050]
    else:
        return text

'''
构造一个适合于SyPy.稀疏索引的数组。
'''
def _make_float_array():
    return array.array(str("f"))

'''
设置文本名字最大长度
'''
def trim_name(text):
    if text and isinstance(text, str):
        return text[:100]
    else:
        return text

'''
品牌名最大长度
'''
def trim_brand_name(text):
    if text and isinstance(text, str):
        return text[:100]
    else:
        return text



'''
公共类
'''


'''
定义不同模型
'''
# dtype = DTYPE
# class Model:
#     def __init__(self):
#         self.sess = None
#         self.X = None
#         self.y = None
#         self.layer_keeps = None
#         self.vars = None
#         self.keep_prob_train = None
#         self.keep_prob_test = None
#
#     # run model
#     def run(self, fetches, X=None, y=None, mode='train'):
#             # 通过feed_dict传入数据
#             feed_dict = {}
#             if type(self.X) is list:
#                 for i in range(len(X)):
#                     feed_dict[self.X[i]] = X[i]
#             else:
#                 feed_dict[self.X] = X
#             if y is not None:
#                 feed_dict[self.y] = y
#             if self.layer_keeps is not None:
#                 if mode == 'train':
#                     feed_dict[self.layer_keeps] = self.keep_prob_train
#                 elif mode == 'test':
#                     feed_dict[self.layer_keeps] = self.keep_prob_test
#             #通过session.run去执行op
#             return self.sess.run(fetches, feed_dict)
#
#     # 模型参数持久化
#     def dump(self, model_path):
#         var_map = {}
#         for name, var in self.vars.iteritems():
#             var_map[name] = self.run(var)
#         pkl.dump(var_map, open(model_path, 'wb'))
#         print('model dumped at', model_path)


'''
稀疏矩阵优化，按照行向量拆分
'''
class SparseMarixOptimize(BaseEstimator,TransformerMixin):
    def __init__(self,dtype = np.float32):
        print('稀疏矩阵优化初始化开始...')
        self.dtype = dtype
        print('稀疏矩阵优化初始化结束...\n\n{}\n'.format("*"*200))

    def fit(self,X,*arg):
        print('稀疏矩阵优化开始...')
        return self

    def transform(self,X):
        if X is None:
            raise ValueError("输入值不能为空!")
        print('稀疏矩阵优化结束...\n\n{}\n'.format("*" * 200))
        return sp.csc_matrix(X,dtype = self.dtype)



'''
这是将数值化为向量的类。
'''
class NumericalVectorizer(CountVectorizer):
    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = _make_int_array()
        values = _make_float_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            current_num = 0
            for feature in analyze(doc):
                maybe_float = try_float(feature)
                if maybe_float > 0 and maybe_float <= 200:
                    current_num = maybe_float
                    continue
                try:
                    if current_num == 0:
                        continue
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = current_num / 200
                        current_num = 0
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.float32)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=np.float32)
        X.sort_indices()
        return vocabulary, X

'''
获取最大得分类
'''
'''
获取我们最大得分的类。
'''
class PredictProbaTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, est, target_column):
        self.target_column = target_column
        self.est = est
    '''
    target_column:为我们的目标值。
    '''
    def fit(self, X, y):
        self.est.fit(X, X[self.target_column])
        return self

    def transform(self, X):
        return self.est.predict_proba(X)


'''
数值化为向量的类。
'''
class NumericalVectorizer(CountVectorizer):
    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = _make_int_array()
        values = _make_float_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            current_num = 0
            for feature in analyze(doc):
                maybe_float = try_float(feature)
                if maybe_float > 0 and maybe_float <= 200:
                    current_num = maybe_float
                    continue
                try:
                    if current_num == 0:
                        continue
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = current_num / 200
                        current_num = 0
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.float32)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=np.float32)
        X.sort_indices()
        return vocabulary, X
