# coding=utf-8
import os
import random
import jieba
import re
import string
from hanziconv import HanziConv

fuhao = [u' ', u'。', u'，', u'《', u'》', u'（', u'）', u'【', u'】', u'〈', u'〉', u'；',
         u':', u'：', u'“ ', u'”', u'‘', u'’', u'·', u'—', u'…', u'–', u'．', u'、', u'~',
         u'`', u'.', u'&', u'+', u'=', u'“', u',', u'[', u']', u'<']
punc = "｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
punc = punc.decode("utf-8")


def read_file():
    fp1 = open('weibo/sample_0.1_1', 'r')   # “愤怒”
    fp2 = open('weibo/sample_0.1_2', 'r')   # “厌恶”
    fp3 = open('weibo/sample_0.1_3', 'r')   # “高兴”
    fp4 = open('weibo/sample_0.1_4', 'r')   # “低落”
    fs_neg = open('neg_weibo.txt', 'w')
    fs_pos = open('pos_weibo.txt', 'w')

    positive_weibo = []
    negative_weibo = []
    for line in fp1.readlines():
        negative_weibo.append(line.rstrip())
    for line in fp2.readlines():
        negative_weibo.append(line.rstrip())
    for line in fp4.readlines():
        negative_weibo.append(line.rstrip())
    for line in fp3.readlines():
        positive_weibo.append(line.rstrip())

    # print negative_weibo[4].decode('gbk')
    # print positive_weibo[0].decode('gbk')
    print 'negative_weibo num: ', len(negative_weibo)
    print 'positive_weibo num: ', len(positive_weibo)

    # weibo = []

    for i in range(len(negative_weibo)):
        try:
            negative_weibo[i].decode('gbk')
        except Exception, e:
            continue
        sim_weibo = HanziConv.toSimplified(negative_weibo[i].decode('gbk'))
        neg_weibo = preprocess_weibo(sim_weibo)
        seg_list = jieba.lcut(neg_weibo)
        seg_list = [w for w in seg_list if w not in punc and w not in fuhao and w not in string.punctuation]
        each_weibo = ' '.join(seg_list)
        fs_neg.write('%s %d\n' % (each_weibo.encode('utf-8'), 0))
        print i
        # weibo.append(seg_list)
        # print seg_list

    for j in range(len(positive_weibo)):
        try:
            positive_weibo[j].decode('gbk')
        except Exception, e:
            continue
        sim_weibo = HanziConv.toSimplified(positive_weibo[i].decode('gbk'))
        pos_weibo = preprocess_weibo(sim_weibo)
        seg_list = jieba.lcut(pos_weibo)
        seg_list = [w for w in seg_list if w not in fuhao and w not in punc and w not in string.punctuation]
        each_weibo = ' '.join(seg_list)
        fs_pos.write('%s %d\n' % (each_weibo.encode('utf-8'), 1))
        print j

    fp1.close()
    fp2.close()
    fp3.close()
    fp4.close()
    fs_neg.close()
    fs_pos.close()


def preprocess_weibo(text):
    text0 = (u'据@中警安徽 ，14日傍晚，安徽阜阳有好心人见两个小女孩在街上游荡，随后报警。@颍东公安在线 民警询问女孩，'
             u'得知女孩的奶奶重男轻女，在女孩父母离婚后，便想赶走两女孩。民警只好联系女孩的母亲，母亲表示今后她会好好照顾孩子，不再让孩子受委屈。')
    text1 = (u'回覆 :wish u get rich by yourself!!! much meaningful  :i will do what  白大哥 say , '
             u'24000 buy buy buy  thz 白大哥對說：師兄，小弟今早之跌市警告應驗啦')
    text2 = (u'@潘石屹@任志强@王石@余英@倪建达@丁祖昱，这是为什么呢？')
    text3 = (u'...ab cd d e ,, ,, da wq')
    text4 = (u'没有希望的国家 操！  :汽油涨价。。。自焚不起了 :  :   :   :     :不忍看，中国这什么，助纣为虐，必备纣灭！')
    # print text4
    # seg = jieba.lcut(text2)
    # print '/'.join(seg)
    # ^[\u4E00-\u9FFF]+$ match all chinese

    hashtags_re = re.compile(u'@[\u4E00-\u9FFF]+')
    text_re = re.sub(hashtags_re, '_@@@_', text)

    handles_re = re.compile(u'#[\u4E00-\u9FFF]+')
    text_re = re.sub(handles_re, '_###_', text_re)

    url_re = re.compile(r'(http|https|ftp)://[a-zA-Z0-9\./]+')
    text_re = re.sub(url_re, '_URL_', text_re)

    repeat_re = re.compile(r'(.)\1{1,}', re.IGNORECASE)

    def rpt_repl(match):
        return match.group(1)
    text_re = re.sub(repeat_re, rpt_repl, text_re)
    '''
    seg_list = jieba.lcut(text_re)
    print seg_list
    print u':' in fuhao
    seg_list = [w for w in seg_list if w in fuhao]

    print text_re
    print seg_list
    '''
    return text_re

if __name__ == '__main__':
    read_file()
    # preprocess_weibo('')
