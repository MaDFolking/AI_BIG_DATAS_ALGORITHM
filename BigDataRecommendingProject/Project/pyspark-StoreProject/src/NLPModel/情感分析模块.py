#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys
import imp
'''
python3默认sys防止utf-8乱码，所以直接用imp 加载即可。
'''
imp.reload(sys)

'''
用户情感分析业务
主要针对用户评论进行评判好坏评论，以此根据自身业务改善需求。
'''

'''
技术流程:
这个并非是真实公司数据，而且这种评论一般也只有大网站才会有相对多的数据量，在这个项目中这个业务模块我更偏重是当练习使用，
由于NLP这种业务通常维度都不低，而数据量不算很大，所以使用SVM来做。下面说下大概流程:

step1:加载数据和分词，我这里业务简单，直接用jieba.cut分词，然后apply聚合到积极词汇和非积极词汇的第一个坐标中。
如果是比较复杂的情景,需要借助机器学习的HMM,CTR，以及正则表达式。
而jieba分词也是四个模式。
(1)全模式: jieba.cut(str1,cut_all = True)
举例：我/来到/北京/清华/清华大学/华大/大学 所有可能性都分出。
(2)精确模式: 默认
举例:我/来到/北京/清华大学
(3)新闻识别:jieba.analyse.extract_tags(str1,2) ：参数setence对应str1为待提取的文本,topK对应2为返回几个TF/IDF权重最大的关键词，默认值为20
他来到了网易杭研大厦。 杭研不是单词，但能被这个Viterbi算法识别。
(4)搜索引擎模式: jieba.cut_for_search(str3)
类似全模式，但内容更多，把搜索组合都排列。在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
召回率:检索的文档数/相关文档总数

step2:
如果词性是英文且特别复杂，需要以下俩个选择操作
(1))Stemming词干提取:简单说就是把不影响词性的inflection的小尾巴砍掉。
walking = walk
(2)Lemmatization词形归一: 把各种类型的词的变形，都归为⼀个形式went 归一成 go are 归一成 be
但有个缺点:但went为动词时，是go的过去式，名词是人名。所以需要POS Tag来判断该单词的词性。

step3:如果需要，创建停止词。以上基本操作完事后，需要将目标数据保存成二进制文件，方便我们后面预测时使用

step4:生成词向量，取出词的重要性，通常可以用平均法，或者tf-idf,不推荐卡方。

step5:Word2Vec训练词向量,并保存生成后的模型，最后用测试集训练并保存其模型

step6:加载保存后的数据

step7:SVM训练，并保存模型

step8:构建待预测的词向量。加载上面训练测试集的词向量,并用上面的词向量函数进行转换

step9:情感分析:用jieba.lcut进行重新切分，用step8中作为predict的目标值，加载svm模型进行预测。

step10:自行测试

PS:最后一定要生成word_list列表再进行机器学习处理
'''

'''
技术说明:

1.jieba分词原理:基于Trie树结构实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图（DAG)
采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
对于未登录词，采用了基于汉字成词能力的HMM模型，使用了Viterbi算法

2.jieba分词过程:
加载字典, 生成trie树
给定待分词的句子, 使用正则获取连续的 中文字符和英文字符, 切分成 短语列表, 对每个短语使用DAG(查字典)和动态规划,
 得到最大概率路径, 对DAG中那些没有在字典中查到的字, 组合成一个新的片段短语, 使用HMM模型进行分词, 也就是作者说的识别新词, 即识别字典外的新词.
使用python的yield 语法生成一个词语生成器, 逐词语返回. 当然, 我认为直接返回list, 效果也差不到哪里去.

3.word2vec:
(1)用途:
将自然语言中的字词转化为计算机可以理解的稠密向量。避免了之前One-Hot处理字词的维度灾难。
将One-Hot Encorder转化为低纬度的连续值，其中意思相近的词
被映射到向量空间中相近的位置。

(2)原理;
本质是个简单化的神经网络。
输入:One-Hot Vector. 进入hidden层后，hidden是没有激活函数的，也就是线性单元。
而OutputLayer维度跟InputLayer维度一样，用SoftMax回归。
我们的dense vector就是hidden层的输出单元。同时设置了权重。

(3)模式:
分CBOW和Skip-Gram
CBOW:从原始语句推测出目标字词,适用于小型语料。
Skip-Gram:从目标字词推测出原始语句，适用于大型语料。
举例:
对同样一个句子：Hangzhou is a nice city。我们要构造一个语境与目标词汇的映射关系，其实就是input与label的关系。
这里假设滑窗尺寸为1
CBOW可以制造的映射关系为：[Hangzhou,a]—>is，[is,nice]—>a，[a,city]—>nice
Skip-Gram可以制造的映射关系为(is,Hangzhou)，(is,a)，(a,is)， (a,nice)，(nice,a)，(nice,city)

(4)训练优化:
到这里，你可能会注意到，这个训练过程的参数规模非常巨大。
假设语料库中有30000个不同的单词，hidden layer取128，word2vec两个权值矩阵维度都是[30000,128]，
在使用SGD对庞大的神经网络进行学习时，将是十分缓慢的。而且，你需要大量的训练数据来调整许多权重，
避免过度拟合。数以百万计的重量数十亿倍的训练样本意味着训练这个模型将是一个野兽。
一般来说，有Hierarchical Softmax、Negative Sampling等方式来解决。

(5)参数讲解:
sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。就是一个向量的维度。
window：表示当前词与预测词在一个句子中的最大距离是多少
alpha: 是学习速率
seed：用于随机数发生器。与初始化词向量有关。
min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
workers参数控制训练的并行数。
hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
hashfxn： hash函数来初始化权重。默认使用python的hash函数
iter： 迭代次数，默认为5
trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
batch_words：每一批的传递给线程的单词的数量，默认为1000


'''

'''
一些技巧:
字典打分:
dics = {}
total_score = sum(dics.get(word,0) for word in words)
遇到新闻词或者更深词如何解决
用机器学习贝叶斯分类
'''

'''
PS:情感分析是个很复杂的技术，后续随着经验和个人的学习，还会更新这里。
PS:为何用One-Hot编码:
除了要转成数字便于计算机识别外。我们每次都是在词库中学习一个单词，也就是取一个单词，如果不转为0-1向量，速度相当慢。
转成后每次学习都时判断0，1而已，速度非常快。
'''

'''
加载数据,并做分词，拆分训练集和数据集
np.concatenate:沿着当前维度拼接。我们将正评论都设置为1，负面评论都设置为0，进行拼接。
我们的数据都是每一行一个评论，所以将所有行的第一个数都聚合到分词即可： pos[0].apply(cw)
最后将分词后的数据和设置为1，0的评论进行分割。
将数组保存为NumPy .npy格式的二进制文件。 =>numpy.save,最后需要用此来做比较
优化点:如果你的业务多次分词地方，最好放在一个临时变量里，直接调用。
'''
def load_data():
    neg = pd.read_csv('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\neg.xls',header = None)
    pos = pd.read_csv('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\pos.xls',header = None)
    cw = lambda x:list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    comment = np.concatenate((np.ones(len(pos)),np.zeros(len(neg))))

    X_train,X_test,y_train,y_test = train_test_split(np.concatenate((pos['words'],neg['words'])),comment,test_size=0.2)
    np.save('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\y_train.npy',y_train)
    np.save( 'D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\y_test.npy',y_test)

    return X_train,x_test

'''
生成词向量，也是采取词汇的重要性。
可以采用tf形式，但这个样本数据太少，我认为没必要。
我们这里采用比较暴力的方法，将每个句子的所有词向量取均值，来生成一个句子的vector
因为每一行只有1个，需要reshape((1,size)),创建称size维的1维数组，这是我们通常评论语句的格式。
count是float
'''

def parse_vector(text,size,word):
    vec = np.zeros(size).reshape((1,size))
    count = 0.
    for i in text:
        try:
            vec += word[i].reshape((1,size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

'''
Word2Vec训练词向量
通常我们设置一个300维度的词向量，工业中通常是300或者500维度
初始化的Word2Vec需要在训练集和测试集分别训练。
然后需要进行concatenate拼接,里面被parse_vector转换。
'''
def get_vec(X_train,X_test):
    nDim = 300
    word_parse = Word2Vec(size = n_dim,min_count=10)
    word_parse.build_vocab(X_train)
    word_parse.train(X_train)
    train_vec = np.concatenate(([parse_vector(i,nDim,word_parse) for i in X_train]))
    np.save('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\train_vec.npy',train_vec)
    print(train_vec.shape)
    word_parse.train(X_test)
    word_parse.save('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\model\\w2v_model.pkl')
    test_vec = np.concatenate(([parse_vector(i,nDim,word_parse) for i in X_test]))
    np.save('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\test_vec.npy',test_vec)
    print(test_vec.shape)

'''
加载向量和训练测试集的目标数据
'''
def get_data():
    train_vec = np.load('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\train_vec.npy')
    test_vec = np.load('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\test_vec.npy')
    y_train = np.load('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\y_train.npy')
    y_test = np.load('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\svm_data\\y_test.npy')
    return train_vec, y_train, test_vec, y_test

'''
训练SVM模型
训练训练集的词向量和目标值
'''
def svm_train(train_vec,y_train,test_vec,y_test):
    model = SVC(kernel='rbf',verbose=True)
    model.fit(train_vec,y_train)
    joblib.dump(clf, 'D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\model\\model.pkl')
    print(model.score(test_vec, y_test))

'''
构建待预测句子的向量
'''
def get_predict_vec(words):
    nDim = 300
    word_parse = Word2Vec.load('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\model\\w2v_model.pkl')
    train_vec = parse_vector(words,nDim,word_parse)
    return train_vec

'''
情感分析函数
jieba.lcut 直接返回list
'''
def svm_predict(string):
    words = jieba.lcut(string)
    words_vec = get_predict_vec(words)
    clf = joblib.load('D:\\自然语言\\自然语言小案例\\Chinese-sentiment-analysis\\Chinese-sentiment-analysis\\data\\model\\model.pkl')
    result = clf.predict(words_vec)

    if int(result[0]) == 1:
        print(string,'positive')
    else:
        print(string,'negative')

'''
测试:具体不写了，一一调用函数即可。
'''
def main():
    string1 = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    string2 = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(string1)
    svm_predict(string2)

if __name__ == '__main__':
    main()

















