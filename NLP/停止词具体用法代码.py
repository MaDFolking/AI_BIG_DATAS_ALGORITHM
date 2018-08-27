import sys
import jieba
from os import path

d = path.dirname(__file__)
PATH = 'D:\\kaggle比赛\\新闻分类项目\\data_jieba\\'
stop_words_path = PATH + 'stop_words.txt'
text_path = PATH + 'test_text.txt'

text = open(path.join(d,text_path)).read()



def jiebaclearText(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr = "/ ".join(seg_list)
    f_stop = open(stop_words_path)
    try:
        f_stop_text = f_stop.read()
    finally:
        f_stop.close()
    f_stop_seg_list = f_stop_text.split('\n')
    for myword in liststr.split('/'):
        if not (myword.strip() in f_stop_seg_list) and len(myword.strip()) > 1:
            mywordlist.append(myword)
    return ''.join(mywordlist)


text1 = jiebaclearText(text)

print(text1)
