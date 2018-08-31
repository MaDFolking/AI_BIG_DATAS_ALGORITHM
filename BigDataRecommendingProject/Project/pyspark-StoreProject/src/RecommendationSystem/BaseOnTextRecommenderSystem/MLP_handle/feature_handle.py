
import re
import array
import numpy as np
import pandas as pd
import util_model as util

from tqdm import tqdm
from scipy import sparse
from functools import partial
from scipy import sparse as sp
from collections import defaultdict
from scipy.sparse import csr_matrix
from multiprocessing.pool import Pool
from sklearn.base import TransformerMixin
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.feature_extraction.text import _make_int_array, CountVectorizer
from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one, _name_estimators
from util_model import extract_year,has_digit,try_float,clean_text,UNK,trim_description,trim_brand_name,trim_name


'''
该模块是特征工程使用的类以及数据处理使用的类。
'''

'''
数据处理模块
'''
'''
1.数据预处理
'''
class PreprocessDataKL(BaseEstimator, TransformerMixin):

    def __init__(self, num_brands, repl_patterns):
        print("数据预处理初始化开始...")
        self.num_brands = num_brands
        self.repl_patterns = repl_patterns
        print("数据预处理初始化结束...\n\n{}\n".format("*"*200))

    '''
    拟合品牌数量
    '''
    def fit(self, X, y = None):
        print("数据预处理开始...")
        self.pop_brands = X['brand_name'].value_counts().index[:self.num_brands]
        return self
    '''
    nan值都设定为unkown 如果是中文根据业务来分。
    '''
    def transform(self, X):
        if X is None:
            raise ValueError("输入值不能为空!")
        X['category_name'] = X['category_name'].fillna('unknown').map(str)
        X['brand_name'] = X['brand_name'].fillna('unknown').map(str)
        X['item_description'] = X['item_description'].fillna('').map(str)
        X['name'] = X['name'].fillna('').map(str)

        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        X.loc[~X['brand_name'].isin(self.pop_brands), 'brand_name'] = 'Other'
        X['category_name_l1'] = X['category_name'].str.split('/').apply(lambda x: x[0])
        X['category_name_l1s'] = \
            X['category_name'].str.split('/').apply(
                lambda x: x[0] if x[0] != 'Women' else '/'.join(x[:2]))
        X['category_name_l2'] = \
            X['category_name'].str.split('/').apply(lambda x: '/'.join(x[:2]))
        for pat, repl in self.repl_patterns:
            X['item_description'] = X['item_description'].str.replace(
                pat, repl, flags=re.IGNORECASE)

        no_description = X['item_description'] == 'No description yet'
        X.loc[no_description, 'item_description'] = ''
        X['no_description'] = no_description.astype(str)
        X['item_condition_id'] = X['item_condition_id'].map(str)
        X['shipping'] = X['shipping'].map(str)
        print("数据预处理结束...\n\n{}\n".format("*"*200))
        return X

'''
2.FastTokenizer 转化为词向量
FastTokenizer并行处理
'''
class PreprocessDataPJ(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=4, hashchars=False, stem=True):
        print("FastTokenizer转为词向量初始化开始...")
        self.n_jobs = n_jobs
        self.hashchars = hashchars
        self.stem = stem
        print("FastTokenizer转为词向量初始化结束...\n\n{}\n".format("*"*200))

    def fit(self, X, y = None):
        print("FastTokenizer转为词向量开始...")
        return self
    '''
    典型的，函数在执行时，要带上所有必要的参数进行调用。
    然后，有时参数可以在函数被调用之前提前获知。这种情况下，
    一个函数有一个或多个参数预先就能用上，以便函数能用更少的参数进行调用。
    '''
    def transform(self, X):
        tokenizer = FastTokenizer()
        '''
        partital:第一个参数是个函数，意思是我们用后面的参数当做是第一个参数也就是这个函数形参去调用这个函数。
        我们这里是清理文本异常的函数：clean_text
        '''
        clean_text_ = partial(clean_text, tokenizer=tokenizer, hashchars=self.hashchars)
        '''
        生成词向量后，开始一系列操作。
        '''
        X['item_condition_id'] = X['item_condition_id'].fillna('UNK').astype(str)
        X['shipping'] = X['shipping'].astype(str)
        X['item_description'][X['item_description'] == 'No description yet'] = UNK
        X['item_description'] = X['item_description'].fillna('').astype(str)
        X['name'] = X['name'].fillna('').astype(str)

        # trim
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        '''
        Pool:相当于java的线程池，我们这厮用4个平法处理。tqdm是进程可视化。
        tqdm: https://blog.csdn.net/langb2014/article/details/54798823?locationnum=8&fps=1
        '''
        if self.stem:
            with Pool(4) as pool:
                X['name_clean'] = pool.map(clean_text_, tqdm(X['name'], mininterval=2), chunksize=1000)
                X['desc_clean'] = pool.map(clean_text_, tqdm(X['item_description'], mininterval=2), chunksize=1000)
                X['brand_name_clean'] = pool.map(clean_text_, tqdm(X['brand_name'], mininterval=2), chunksize=1000)
                X['category_name_clean'] = pool.map(clean_text_, tqdm(X['category_name'], mininterval=2),
                                                    chunksize=1000)
        X['no_cat'] = X['category_name'].isnull().map(int)
        cat_def = [UNK, UNK, UNK]
        X['cat_split'] = X['category_name'].fillna('/'.join(cat_def)).map(lambda x: x.split('/'))
        X['cat_1'] = X['cat_split'].map(
            lambda x: x[0] if isinstance(x, list) and len(x) >= 1 else cat_def).str.lower()
        X['cat_2'] = X['cat_split'].map(
            lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else cat_def).str.lower()
        X['cat_3'] = X['cat_split'].map(
            lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else cat_def).str.lower()
        X['is_bundle'] = (X['item_description'].str.find('bundl') >= 0).map(int)

        print("FastTokenizer转为词向量结束...\n\n{}\n".format("*" * 200))
        return X

'''
3.处理敏感词
品牌商标的敏感词:
比如1-9这种型号的， as at X10 X13这种敏感词，我们需要脱敏处理。
中文里，也必须熟悉品牌的词汇才可以做。
'''
class ExtractSpecifics(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        print("敏感词处理开始...")
        return self

    def transform(self, X):
        keys = {"1", "2", "3", "4", "5", "7", "8", "9", "a", "as", "at", "b", "bars", "beautiful",
                "boots", "bottles", "bowls", "box", "boxes", "brand", "bras", "bucks",
                "cans", "card", "cards", "case", "cm", "comes",
                "compartments", "controllers", "cream", "credit", "crop", "dd",
                "dollar", "dollars", "dolls", "dress", "dvds", "each", "edition", "euc",
                "fashion", "feet", "fits", "fl", "ft", "g", "games", "gb",
                "gms", "gold", "gram", "grams",
                "hr", "hrs", "in", "inch", "inches", "k",
                "karat", "layers", "up",
                "meter", "mil", "mini", "mint", "ml", "mm", "month", "mugs", "no", "not", "nwt", "off",
                "onesies", "opi", "ounce", "ounces", "outfits", "oz", "packages", "packets", "packs", "pair", "panels",
                "pants", "patches", "pc", "pics", "piece", "pieces", "pokémon",
                "pokemon", "pounds", "price", "protection", "random", "retro", "ring", "rings", "rolls",
                "samples", "sandals", "series", "sets", "sheets", "shirts", "shoe", "shoes",
                "shows", "slots", "small", "so", "some", "stamped", "sterling", "stickers", "still", "stretch",
                "strips", "summer", "t", "tags", "tiny", "tone", "tubes", "victoria", "vinyl", "w", "waist",
                "waistband", "waterproof", "watt", "white", "wireless", "x10", "x13", "x15", "x3", "x4", "x5", "x6",
                "x7", "x8", "x9", "yrs", "½", "lipsticks", "bar", "apple", "access", "wax", "monster", "spell",
                "spinners", "lunch", "ac", "jamberry", "medal", "gerard"}
        regex = re.compile("(\d+)[ ]?(\w+)", re.IGNORECASE)

        specifics = []
        for x in X:
            spec = {}
            '''
            re.IGNORECASE的意思就是忽略大小写。
            '''
            for val, key in regex.findall(str(x), re.IGNORECASE):
                if key in keys:
                    val = try_float(val)
                    if val > 3000:
                        continue
                    spec[key] = val
                    spec['{}_{}'.format(key, val)] = 1
            specifics.append(spec)
        print("敏感词处理结束...\n\n{}\n".format("*" * 200))
        return specifics



'''
特征工程模块
'''

'''
特征构造:
(1)描述特征-构造该特征
'''
class FeaturesEngItemDescription(BaseEstimator , TransformerMixin):
    def fit(self,X,*arg):
        print('描述特征-构造该特征开始...')
        return self

    def transform(self,X):
        print('描述特征-构造该特征转化开始...')
        if X is None:
            raise ValueError("输入值不能为空...")
        out = pd.DataFrame()
        out['lt'] = np.log1p(X['desc_clean'].str.len()) / 100.0           # 提取长度
        out['yd'] = X['desc_clean'].map(str).map(extract_year) / 2000.0   # 提取年份
        out['nw'] = np.log1p(X['desc_clean'].str.split().map(len))
        print('描述特征-构造该特征结束...\n\n{}\n'.format("*" * 200))
        return out.values

'''
特征构造:
(2)名称特征-构造该特征
'''
class FeaturesEngName(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        print('名称特征-构造该特征开始...')
        return self

    def transform(self, X):
        if X is None:
            raise ValueError("输入值不能为空!")
        out = pd.DataFrame()
        out['lt'] = np.log1p(X['name_clean'].str.len()) / 100.0
        out['yd'] = X['name_clean'].map(str).map(extract_year) / 2000.0
        out['nw'] = np.log1p(X['name_clean'].str.split().map(len))
        print('名称特征-构造该特征结束...\n\n{}\n'.format("*" * 200))
        return out.values

'''
特征分类。
在这里用了并行化处理并返回优化的稀疏矩阵。
'''
class FeaturesPatterns(BaseEstimator, TransformerMixin):
    patterns = [
        "([0-9.]+)[ ]?\$",
        "\$[ ]?([0-9.]+)",
        "paid ([0-9.]+)",
        "bought for (\d+)"
        "(10|14|18|24) gold",
        "of (\d+) ",
        " (\d+) ship",
        "is for all (\d+)",
        "is for (\d+)",
        "firm for (\d+)",
        "bundl \w+ (\d+) ",
        "(\d+) in 1",
        "^(\d+)",
        "\d+ for (\d+)",
        " x(\d+)",
        "\b(\d+)x\b",
        "(\d+)% left",
        "(\d+)[ ]?lipstick",
    ]

    def __init__(self, column):
        print('特征分类初始化开始...\n\n{}\n'.format("*" * 200))
        self.column = column
        print('特征分类初始化结束...\n\n{}\n'.format("*" * 200))

    def fit(self, X, *arg):
        print('特征分类开始...')
        return self

    '''
    Tqdm : 进度条
    '''
    def transform(self, X):
        if X is None:
            raise ValueError("输入值不能为空!")
        cols = []
        self.features_names = []
        X_ = X[self.column].map(lambda x: "" if has_digit(x) else x)
        for pattern_name in tqdm(self.patterns):
            new_col = 'regex_pattern_{}_{}'.format(pattern_name, self.column)
            '''
            并行化处理。
            str.extract: https://blog.csdn.net/claroja/article/details/64929819
            对于系列中的每个主题字符串，从正则表达式pat的第一个匹配中提取组。
            根据业务，不能取超过2000的价格物品，然后都log1p化。最后放入到特征表中。
            返回稀疏矩阵csr_matrix
            '''
            raw_val = X_.str.extract(pattern_name, expand=False).fillna(0)
            if isinstance(raw_val, pd.DataFrame):
                raw_val = raw_val.iloc[:, 0]
            '''
            全部转化为float处理，因为是价格。
            '''
            X[new_col] = raw_val.map(try_float)
            X[X[new_col] > 2000] = 0
            X[new_col] = np.log1p(X[new_col])
            cols.append(new_col)
            self.features_names.append(new_col)
        print('特征分类结束...\n\n{}\n'.format("*" * 200))
        return csr_matrix(X.loc[:, cols].values)

    def get_feature_names(self):
        return self.features_names

'''
特征过滤和转换
(1)多线程处理
'''
class PreprocessDataPJ(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=4, hashchars=False, stem=True):
        print("特征过滤和转换多线程初始化开始...")
        self.n_jobs = n_jobs
        self.hashchars = hashchars
        self.stem = stem
        print("特征过滤和转换多线程初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, y):
        print("特征过滤和转换多线程开始...")
        return self
    '''
    add是个求和函数，partial用法如下
    In [12]: plus = partial(add,100)
    In [13]: plus(9)
    Out[13]: 109
    '''
    def transform(self, X):
        tokenizer = FastTokenizer()
        clean_text_ = partial(clean_text, tokenizer=tokenizer, hashchars=self.hashchars)
        X['item_condition_id'] = X['item_condition_id'].fillna('UNK').astype(str)
        X['shipping'] = X['shipping'].astype(str)
        X['item_description'][X['item_description'] == 'No description yet'] = UNK
        X['item_description'] = X['item_description'].fillna('').astype(str)
        X['name'] = X['name'].fillna('').astype(str)
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        '''
        map(func, iterable[, chunksize])
        map方法与在功能上等价与内置的map()，只不过单个任务会并行运行。它会使进程阻塞直到结果返回。
        但需注意的是其第二个参数虽然描述的为iterable, 但在实际使用中发现只有在整个队列全部就绪后，程序才会运行子进程。

        tqdm第一个参数必须是迭代器，
        mininterval:
        可选浮动 最小进度显示更新间隔，以秒为单位[默认值：0.1 ]。最大间隔：浮动，可选
        最大进度显示更新间隔，以秒为单位[默认值：10 ]。
        自动调整'MITETIES ''对应'min间隔''长时间显示后更新滞后。只工作“动态微小”或启用监视器线程。
        tqdm(X['name'], mininterval=2),
        tqdm(X['item_description'], mininterval=2),
        tqdm(X['brand_name'], mininterval=2),
        tqdm(X['category_name'], mininterval=2),
        '''
        if self.stem:
            with Pool(4) as pool:
                X['name_clean'] = pool.map(clean_text_, tqdm(X['name'], mininterval=2), chunksize=1000)
                X['desc_clean'] = pool.map(clean_text_, tqdm(X['item_description'], mininterval=2), chunksize=1000)
                X['brand_name_clean'] = pool.map(clean_text_, tqdm(X['brand_name'], mininterval=2), chunksize=1000)
                X['category_name_clean'] = pool.map(clean_text_, tqdm(X['category_name'], mininterval=2),
                                                    chunksize=1000)
        X['no_cat'] = X['category_name'].isnull().map(int)
        cat_def = [UNK, UNK, UNK]
        X['cat_split'] = X['category_name'].fillna('/'.join(cat_def)).map(lambda x: x.split('/'))
        X['cat_1'] = X['cat_split'].map(
            lambda x: x[0] if isinstance(x, list) and len(x) >= 1 else cat_def).str.lower()
        X['cat_2'] = X['cat_split'].map(
            lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else cat_def).str.lower()
        X['cat_3'] = X['cat_split'].map(
            lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else cat_def).str.lower()
        X['is_bundle'] = (X['item_description'].str.find('bundl') >= 0).map(int)

        print("特征过滤和转换多线程结束...\n\n{}\n".format("*" * 200))
        return X

'''
特征过滤
(2)KL处理
'''
class PreprocessDataKL(BaseEstimator, TransformerMixin):

    def __init__(self, num_brands, repl_patterns):
        print("特征过滤和转换KL处理初始化开始...")
        self.num_brands = num_brands
        self.repl_patterns = repl_patterns
        print("特征过滤和转换KL处理初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, y):
        print("特征过滤和转换KL处理开始...")
        self.pop_brands = X['brand_name'].value_counts().index[:self.num_brands]
        return self

    def transform(self, X):
        # fill missing values
        X['category_name'] = X['category_name'].fillna('unknown').map(str)
        X['brand_name'] = X['brand_name'].fillna('unknown').map(str)
        X['item_description'] = X['item_description'].fillna('').map(str)
        X['name'] = X['name'].fillna('').map(str)

        # trim
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        X.loc[~X['brand_name'].isin(self.pop_brands), 'brand_name'] = 'Other'
        X['category_name_l1'] = X['category_name'].str.split('/').apply(lambda x: x[0])
        X['category_name_l1s'] = \
            X['category_name'].str.split('/').apply(
                lambda x: x[0] if x[0] != 'Women' else '/'.join(x[:2]))
        X['category_name_l2'] = \
            X['category_name'].str.split('/').apply(lambda x: '/'.join(x[:2]))
        for pat, repl in self.repl_patterns:
            X['item_description'] = X['item_description'].str.replace(
                pat, repl, flags=re.IGNORECASE)

        no_description = X['item_description'] == 'No description yet'
        X.loc[no_description, 'item_description'] = ''
        X['no_description'] = no_description.astype(str)
        X['item_condition_id'] = X['item_condition_id'].map(str)
        X['shipping'] = X['shipping'].map(str)
        print("特征过滤和转换KL处理结束...\n\n{}\n".format("*" * 200))
        return X

'''
特征过滤
(3)稀疏矩阵过滤器。
设置价格范围的类。
'''
class SparsityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_nnz=None):
        print("稀疏矩阵过滤器初始化开始...")
        self.min_nnz = min_nnz
        print("稀疏矩阵过滤器初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, y = None):
        print("稀疏矩阵过滤器开始...")
        self.sparsity = X.getnnz(0)
        return self

    def transform(self, X):
        if X is None:
            raise ValueError("特征过滤处理失败...")
        print("稀疏矩阵过滤器结束...\n\n{}\n".format("*" * 200))
        return X[:, self.sparsity >= self.min_nnz]

'''
特征过滤
(2)过滤假冒产品，利用英文敏感词 for like fits 中文同理，比如中国乔丹 美国乔丹这种格式。
'''
class FalseBrands(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        print("过滤假冒产品开始...")
        return self

    def false_brand_detector(self, prefix):
        def helper(row):
            return prefix + ' ' + str(row.brand_name).lower() in str(row.item_description).lower()

        return helper

    def transform(self, X):
        if X is None:
            raise ValueError("过滤假冒产品处理失败...")
        print("过滤假冒产品结束...\n\n{}\n".format("*" * 200))
        return pd.DataFrame({
            'for_brand': X.apply(self.false_brand_detector('for'), axis=1),
            'like_brand': X.apply(self.false_brand_detector('like'), axis=1),
            'fits_brand': X.apply(self.false_brand_detector('fits'), axis=1)
        })

'''
特征拼接(拼接文本)
每行都用cs001,cs002来拼接。
'''
class ConcatTexts(BaseEstimator, TransformerMixin):
    def __init__(self, columns, use_separators=True, output_col='text_concat'):
        print("特征拼接初始化开始...")
        self.use_separators = use_separators
        self.columns = columns
        self.output_col = output_col
        print("特征拼接初始化结束...\n\n{}\n".format("*"*200))

    def fit(self, X, *args):
        print("特征拼接开始...")
        return self

    def transform(self, X):
        if X is None:
            raise ValueError("特征拼接处理失败...")
        X[self.output_col] = ''
        if self.use_separators:
            for i, col in enumerate(self.columns):
                X[self.output_col] += ' cs00{} '.format(i)
                X[self.output_col] += X[col]
        else:
            for i, col in enumerate(self.columns):
                X[self.output_col] += X[col]
        print("特征拼接结束...\n\n{}\n".format("*" * 200))
        return X

'''
填充nan值
'''
class FillEmpty(BaseEstimator, TransformerMixin):
    def fit(self, X, *args):
        print("填充nan值开始...")
        return self

    def transform(self, X):
        X['name'].fillna('unk', inplace=True)
        X['item_condition_id'] = X['item_condition_id'].fillna('unk')
        X['category_name'].fillna('unk', inplace=True)
        X['brand_name'].fillna('unk', inplace=True)
        X['shipping'].fillna(0, inplace=True)
        X['item_description'].fillna('unk', inplace=True)
        print("填充nan值结束...\n\n{}\n".format("*"*200))
        return X

'''
转化为字典类
'''
class PandasToRecords(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        print("转化为字典类开始...")
        return self

    def transform(self, X):
        print("转化为字典类结束...\n\n{}\n".format("*" * 200))
        return X.to_dict(orient='records')

'''
优化器类，矩阵按列
'''
class SparseMatrixOptimize(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=np.float32):
        print("矩阵按列优化初始化开始...")
        self.dtype = dtype
        print("矩阵按列优化初始化结束...\n\n{}\n".format("*" * 200))

    def fit(self, X, *arg):
        print("矩阵按列优化开始...")
        return self

    def transform(self, X):
        print("矩阵按列优化结束...\n\n{}\n".format("*" * 200))
        return sp.csr_matrix(X, dtype=self.dtype)

'''
切割矩阵最大值。这个类目的在于选取每行最大值。为后期分析做准备。
np.nanmax: 沿轴返回数组的最大值或最大值，忽略任何NaN。当遇到全NaN切片时，RuntimeWarning会引发a并为该切片返回NaN。
'''
class SanitizeSparseMatrix(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        print("切割矩阵最大值开始...")
        self.datamax = np.nanmax(X.data)
        return self

    def transform(self, X):
        X.data[np.isnan(X.data)] = 0
        X.data = X.data.clip(0, self.datamax)
        print("切割矩阵最大值结束...\n\n{}\n".format("*" * 200))
        return X

'''
FastWord初始化类，里面设置这次我们提取字符的范围，方便后面FastWord调用
'''
class FastTokenizer():
    print("FastWord初始化类开始...")
    _default_word_chars = \
        u"-&" \
        u"0123456789" \
        u"ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
        u"abcdefghijklmnopqrstuvwxyz" \
        u"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß" \
        u"àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ" \
        u"ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğ" \
        u"ĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł" \
        u"ńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧ" \
        u"ŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ" \
        u"ΑΒΓΔΕΖΗΘΙΚΛΜΝΟΠΡΣΤΥΦΧΨΩΪΫ" \
        u"άέήίΰαβγδεζηθικλμνξοπρςστυφχψω"
    '''
    去重
    '''
    _default_word_chars_set = set(_default_word_chars)
    '''
    设置可能出现的隔断
    '''
    _default_white_space_set = set(['\n','\t',' '])
    '''
    将内容放入列表中
    '''
    def __call__(self, text: str):
        tokens = []
        for ch in text:
            if len(tokens) == 0:
                tokens.append(ch)
                continue
            if self._merge_with_prev(tokens, ch):
                tokens[-1] = tokens[-1] + ch
            else:
                tokens.append(ch)
        print("FastWord初始化类结束...\n\n{}\n".format("*" * 200))
    '''
    判断我们的词是否属于我们设置的英文单词中或者我们设置的间隔符。如果在，调用上限方法中，增加这个字符。
    '''
    def _merge_with_prev(self, tokens, ch):
        return (ch in self._default_word_chars_set and tokens[-1][-1] in self._default_word_chars_set) or \
               (ch in self._default_white_space_set and tokens[-1][-1] in self._default_white_space_set)


'''
打印矩阵
'''
class ReportShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        print("打印矩阵开始...")
        return self

    def transform(self, X):
        logger.info('=' * 30)
        logger.info("矩阵形状为 {} 最小值为 {} 最大值为 {}".format(X.shape, X.min(), X.max()))
        logger.info('=' * 30)
        print("F打印矩阵结束...\n\n{}\n".format("*" * 200))
        return X

'''
特征选择
(1)这里自然不像数据挖掘用方差。
我们这次用SGD稀疏矩阵来选择特征，因为针对的是稀疏矩阵，后续我们会在PipLine管道中先进行PCA分解/SVD分解
'''
class SGDFeatureSelectionV2(BaseEstimator, TransformerMixin):
    def __init__(self, percentile_cutoff=None):
        print("特征选择SGD算法初始化开始...")
        self.percentile_cutoff = percentile_cutoff
        print("特征选择SGD算法初始化结束...\n\n{}\n".format("*"*200))

    def fit(self, X, y, *args):
        print("特征选择SGD算法开始...")
        sgd = SGDRegressor(penalty='l1', loss='squared_loss', alpha=3.0e-11, power_t=-0.12, eta0=0.019, random_state=0,
                           average=True)
        sgd.fit(X, np.log1p(y))
        coef_cutoff = np.percentile(np.abs(sgd.coef_), self.percentile_cutoff)
        self.features_to_keep = np.where(np.abs(sgd.coef_) >= coef_cutoff)[0]
        return self

    def transform(self, X):
        if X is None:
            raise ValueError("特征选择SGD算法处理失败...")
        print("特征选择SGD算法结束...\n\n{}\n".format("*" * 200))
        return X[:, self.features_to_keep]

'''
特征选择
（2）拼接并选择特征
返回的是否为pandas的DataFrame类型或者列表类型，如果长度为1，则返回列表类型，如果不是，返回pandas的DataFrame类型。
'''
class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        print("拼接并选择特征初始化开始...")
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]
        print("拼接并选择特征初始化结束...\n\n{}\n".format("*" * 200))

    def check_condition(self, X, col):
        cond = (self.dtype is not None and X[col].dtype == self.dtype) or (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, X, y=None):
        print("拼接并选择特征开始...")
        return self

    def _check_if_all_columns_present(self, X):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(X.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)
    '''
    不要忘了先检查是否为DataFrame类型，我在 fit transform时总遇到这类错误。
    '''
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise KeyError('输入值X不是一个DataFrame类型')

        selected_cols = []
        for col in X.columns:
            if self.check_condition(X, col):
                selected_cols.append(col)

        '''
        如果选择了列并反转= false，则确保列位于数据文件中。
        '''
        self._check_if_all_columns_present(X)

        '''
        如果只返回1列，则返回矢量，而不是数据文件。
        '''
        print("拼接并选择特征结束...\n\n{}\n".format("*" * 200))
        if len(selected_cols) == 1 and self.return_vector:
            return list(X[selected_cols[0]])
        else:
            return X[selected_cols]

'''
三.特征合并
'''

'''
这是本项目的特殊核心类。也是官网的底層代碼写的。所以是借鉴过来的。
作用：连接多个特征处理对象(变压器，就是包含fit,transform的类)的结果。

该估计器将变压器对象的列表并行地应用于
输入数据，然后连接结果。这对组合是有用的。
几个特征提取机制到一个单一的变压器。
变压器的参数可以使用其名称和参数来设置。
用一个字母“x”分隔的名字。变压器可以完全替换。
将其名称的参数设置为另一个变压器，
或通过设置为“没有”删除。

参数
(1)transformer_list:（字符串，变压器）元组列表
要应用于数据的变压器对象列表。第一
每个元组的一半是变压器的名称。ConcatTexts
(2)N-n_jobs:int，可选
并行运行的作业数（默认值1）。

(3)transformer_weights：字典类型，可选
每个变压器的特征的乘法权重。
键是变压器名，值是权重。
'''
class FeatureUnionMP(_BaseComposition, TransformerMixin):
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        print("特征合并初始化开始...")
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self._validate_transformers()
        print("特征合并初始化结束...\n\n{}\n".format("*"*200))

    '''
    deep，如果为true，将返回此估计器的参数和
    包含估计量的子对象。。
    _get_params:将字符串映射到任意映射到它们的值的参数名称。
    '''
    def get_params(self, deep=True):
        return self._get_params('transformer_list', deep=deep)

    '''
    设置子对象
    '''
    def set_params(self, **kwargs):
        self._set_params('transformer_list', **kwargs)
        return self

    '''
    判断是否可fit/transform/fit_transform
    '''
    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)
        self._validate_names(names)
        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and ""transform. '%s' (type %s) doesn't" %(t, type(t)))

    '''
    生成权重，名称，trans
    '''
    def _iter(self):
        get_weight = (self.transformer_weights or {}).get
        return ((name, trans, get_weight(name))
                for name, trans in self.transformer_list if trans is not None)

    '''
    获取特征名称，用__拼接。
    '''
    def get_feature_names(self):
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not " "provide get_feature_names."% (str(name), type(trans).__name__))
            feature_names.extend([name + "__" + f for f in trans.get_feature_names()])
        return feature_names

    '''
    X是样本数，y是shape
    设置线程池，并发处理
    '''
    def fit(self, X, y=None):
        print("特征合并开始...")
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        with Pool(self.n_jobs) as pool:
            transformers = pool.starmap(_fit_one_transformer,((trans, X[trans.steps[0][1].columns], y) for _, trans, _ in self._iter()))
        self._update_transformer_list(transformers)
        return self

    '''
    安装所有变压器，转换数据并连接结果。
    参数
    ----------
    X：迭代或数组，取决于变压器
    输入数据进行转换。
    Y：数组类，形状（n-样本，…），可选
    监督学习的目标。
    退换商品
    -----
    XYT:阵列状或稀疏矩阵，形状（n-样本，SUMNEXNI分量）
    变压器的HSTAC结果。SuMuxNx组件是
    变压器上的n-分量（输出维数）之和。
    '''
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        with Pool(self.n_jobs) as pool:
            result = pool.starmap(_fit_transform_one,((trans, weight, X[trans.steps[0][1].columns], y) for name, trans, weight in self._iter()))

        if not result:
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        '''
        注意下面的优化方案，按照行拼接后，再转成csr按行压缩矩阵，优化。
        '''
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs


    '''
    由每个变压器单独转换X，连接结果。
    参数
    ----------
    X：迭代或数组，取决于变压器
    输入数据进行转换。

    退换商品
    -----
    XYT:阵列状或稀疏矩阵，形状（n-样本，SUMNEXNI分量）
    变压器的HSTAC结果。SuMuxNx组件是
    变压器上的n-分量（输出维数）之和。
    '''
    def transform(self, X):
        with Pool(self.n_jobs) as pool:
            Xs = pool.starmap(_transform_one, ((trans, weight, X[trans.steps[0][1].columns])
                                               for name, trans, weight in self._iter()))
        if not Xs:
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        print("特征合并结束...\n\n{}\n".format("*" * 200))
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, None if old is None else next(transformers))
            for name, old in self.transformer_list
        ]

'''
从给定的变压器构造一个特征联盟。


这是Stand联盟构造函数的速写，它不需要，
并且不允许命名变压器。相反，他们将被给予
根据它们的类型自动命名。它也不允许加权。

参数
----------
＊变形金刚：估计表
N-Jase:int，可选
并行运行的作业数（默认值1）。
退换商品
-----
F:特征联盟


实例
--------
从SkReln.分解导入PCA，TruncatedSVD
管道进口联合会
A*> > MaxeUnand（pCa（））、TunCeDeVSdd（））
特征联盟（nSub＝1）
变换器列表= [（PCA），
PCA（拷贝＝true，迭代*POWER =‘AUTO’，
n1组件=无，随机化状态＝无，
SvdySover＝“Auto'，ToL＝0，WaleNe= FALSE）
（TuncTead SVD）
TruncatedSVD（算法＝随机化），
n-分量＝2，nsiter＝5，
随机状态=无，ToL＝0）]
变换器权重＝无）
'''
"""Construct a FeatureUnion from the given transformers.

This is a shorthand for the FeatureUnion constructor; it does not require,
and does not permit, naming the transformers. Instead, they will be given
names automatically based on their types. It also does not allow weighting.

Parameters
----------
*transformers : list of estimators

n_jobs : int, optional
    Number of jobs to run in parallel (default 1).

Returns
-------
f : FeatureUnion

Examples
--------
>>> from sklearn.decomposition import PCA, TruncatedSVD
>>> from sklearn.pipeline import make_union
>>> make_union(PCA(), TruncatedSVD())    # doctest: +NORMALIZE_WHITESPACE
FeatureUnion(n_jobs=1,
       transformer_list=[('pca',
                          PCA(copy=True, iterated_power='auto',
                              n_components=None, random_state=None,
                              svd_solver='auto', tol=0.0, whiten=False)),
                         ('truncatedsvd',
                          TruncatedSVD(algorithm='randomized',
                          n_components=2, n_iter=5,
                          random_state=None, tol=0.0))],
       transformer_weights=None)

        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
"""
def make_union_mp(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', 1)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return FeatureUnionMP(_name_estimators(transformers), n_jobs=n_jobs)

'''
这是原官网的写法

def make_pipeline(*steps, **kwargs):
    memory = kwargs.pop('memory', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return Pipeline(_name_estimators(steps), memory=memory)

'''








