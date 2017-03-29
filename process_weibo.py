# coding=utf-8
import os
import random
import jieba
import re
# import string
from hanziconv import HanziConv
import numpy as np
from collections import Counter
import xml.etree.ElementTree as ET

random.seed(1177)

punc = "。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
punc = punc.decode("utf-8")
punc_en = " \"$#@%&'()*+,-./:;<=>[\]^_`{|}~·...._"
punc_en = punc_en.decode('utf-8')

emotoin_pos = [u'[挤眼]', u'[亲亲]', u'[太开心]', u'[哈哈]', u'[酷]', u'[来]', u'[good]', u'[haha]',
               u'[ok]', u'[拳头]', u'[赞]', u'[耶]', u'[微笑]', u'[色]', u'[可爱]', u'[嘻嘻]',
               u'[爱你]', u'[心]', u'[鼓掌]', u'[馋嘴]', u'[抱抱_旧]', u'[发红包]', u'[礼物]', u'[害羞]',
               u'[玫瑰]', u'[威武]', u'[得意地笑]', u'[太阳]', u'[哆啦A梦微笑]', u'[冒个泡]', u'[狂笑]',
               u'[飞个吻]', u'[抱抱]', u'[蛋糕]', u'[兔子]', u'[喵喵]', u'[笑哈哈]', u'[花心]', u'[偷笑]',
               u'[偷乐]', u'[推荐]', u'[音乐]', u'[羊年大吉]', u'[噢耶]', u'[微风]', u'[月亮]', u'[话筒]',
               u'[好喜欢]', u'[好棒]', u'[羞嗒嗒]', u'[给力]', u'[江南style]', u'[鲜花]', u'[好爱哦]',
               u'[好得意]', u'[熊猫]', u'[爱心传递]', u'[哇哈哈]', u'[握手]', u'[做鬼脸]', u'[萌]',
               u'[礼花]', u'[帅]']
emotoin_neg = [u'[生病]', u'[失望]', u'[黑线]', u'[吐]', u'[委屈]', u'[悲伤', u'[衰]', u'[愤怒]',
               u'[感冒]', u'[最差]', u'[NO]', u'[怒骂]', u'[困]', u'[哈欠]', u'[打脸]', u'[笑cry]',
               u'[汗]', u'[泪]', u'[晕]', u'[抓狂]', u'[怒]', u'[doge]', u'[蜡烛]', u'[弱]', u'[睡觉]',
               u'[崩溃]', u'[拜拜]', u'[打哈气]', u'[泪流满面]', u'[哼]', u'[草泥马]', u'[挖鼻]', u'[鄙视]',
               u'[阴险]', u'[可怜]', u'[最右]', u'[挖鼻屎]', u'[悲伤]', u'[疑问]', u'[思考]', u'[浮云]',
               u'[伤心]', u'[囧]', u'[呵呵]', u'[吃惊]', u'[抠鼻屎]', u'[bobo抓狂]', u'[闭嘴]', u'[懒得理你]',
               u'[嘘]', u'[围观]']
emotoin_neu = [u'[a]', u'[b]', u'[c]', u'[d]', u'[e]', u'[f]', u'[g]',
               u'[h]', u'[i]', u'[j]', u'[k]', u'[l]', u'[m]', u'[n]',
               u'[o]', u'[p]', u'[q]', u'[r]', u'[s]', u'[t]',
               u'[u]', u'[v]', u'[w]', u'[x]', u'[y]', u'[z]']


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
        seg_list = [w for w in seg_list if w not in punc and w not in punc_en]
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
        seg_list = [w for w in seg_list if w not in fuhao and w not in punc]
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
    # ^[\u4E00-\u9FFF]+$ match all chinese

    emotion_re = re.compile(u'\[([\u4E00-\u9FFFA-Za-z0-9]+)\]')
    text_re = re.sub(emotion_re, '', text)

    hashtags_re = re.compile(u'@[\u4E00-\u9FFFA-Za-z0-9]+')
    text_re = re.sub(hashtags_re, '', text_re)

    riyu_re = re.compile(u'[\u3040-\u3210\u00c0-\u2e40\uA000-\uFFFF\U00010000-\U0001ffff._]+')
    text_re = re.sub(riyu_re, '', text_re)

    huifu_re = re.compile(u'回复|转发')
    text_re = re.sub(huifu_re, '', text_re)

    handles_re = re.compile(u'#[\u4E00-\u9FFFA-Za-z0-9 ]+#')
    text_re = re.sub(handles_re, '', text_re)

    url_re = re.compile(r'[a-zA-z]+://[a-zA-Z0-9\./]+')
    text_re = re.sub(url_re, '', text_re)

    repeat_re = re.compile(r'([!?a-z])\1{1,}', re.IGNORECASE)

    def rpt_repl(match):
        return match.group(1)
    text_re = re.sub(repeat_re, rpt_repl, text_re)

    reg_eng = re.compile(u'[a-zA-Z0-9@#]+')
    text_re = re.sub(reg_eng, '', text_re)

    return text_re


def find_emotion(infile, outfile):
    # f = open('./data/weibo_train_data.txt', 'r')
    f = open(infile, 'r')
    weibo = []
    f.readline()
    for line in f.readlines():
        try:
            # line = HanziConv.toSimplified(line.strip().split(',')[6].decode('utf-8'))
            line = HanziConv.toSimplified(line.strip().decode('utf-8'))
            line = line.lower()
            weibo.append(line)
            print len(weibo)
        except Exception as e:
            continue
    f.close()
    print len(weibo)
    random.shuffle(weibo)
    emotion = []
    for w in weibo:
        emotion.append(re.findall("(\[.*?\])", w))
    # print len(emotion)
    # print emotion[:10]

    idx = []
    for i, w in enumerate(emotion):
        if w != []:
            idx.append(i)
    print len(idx)

    emotion_clean = []
    for i in idx:
        emotion_clean.append(emotion[i])
    # print len(emotion_clean)
    # print emotion_clean[:10]
    '''
    emotion_dict = dict()
    for em in emotion_clean:
        for ee in em:
            try:
                emotion_dict[ee] += 1
            except Exception, e:
                emotion_dict[ee] = 1
    print len(emotion_dict)

    reg = re.compile(u'^\[[0-9]+\]$')
    for k, v in emotion_dict.items():
        if k not in emotoin_pos and k not in emotoin_neg:
            if emotion_dict[k] <= 500:
                del emotion_dict[k]
    for k, v in emotion_dict.items():
        if re.sub(reg, u'', k) == u'' or k in emotoin_neu:
            del emotion_dict[k]

    print len(emotion_dict)

    print 'pos: ', len(emotoin_pos)
    print 'neg: ', len(emotoin_neg)

    newA = Counter(emotion_dict)
    for k, v in newA.most_common(100):
        print k, v
    '''

    weibo_em = []
    for i in idx:
        weibo_em.append(weibo[i])
    # print len(weibo_em)

    sent = []
    weibo_pos = 0
    weibo_neg = 0
    for e in emotion_clean:
        neg_num = 0
        pos_num = 0
        for i in e:
            if i in emotoin_pos:
                pos_num += 1
            if i in emotoin_neg:
                neg_num += 1
        if pos_num > neg_num + 1:
            sent.append(2)
            weibo_pos += 1
        elif pos_num + 1 < neg_num:
            sent.append(0)
            weibo_neg += 1
        else:
            sent.append(1)
    print len(sent)
    print "weibo_pos: ", weibo_pos
    print "weibo_neg: ", weibo_neg

    jieba.load_userdict('./dict/dict_.txt')
    fw = open(outfile, 'w')
    for i in range(len(sent)):
        if sent[i] != 1:
            weibo_r = preprocess_weibo(weibo_em[i])
            # fw.write('%s\n' % weibo_r.encode('utf-8'))
            seg_list = jieba.lcut(weibo_r)
            seg_list = [w for w in seg_list if w not in punc and w not in punc_en]
            if len(seg_list) > 7:
                weibo = ' '.join(seg_list)
                fw.write('%s,%d\n' % (weibo.encode('utf-8'), sent[i]))
    fw.close()


def process_weibo_deep():
    fp = open('./weibo_emotion/train_data_pos.txt', 'r')
    fn = open('./weibo_emotion/train_data_neg.txt', 'r')
    fo_p = open('./weibo_emotion/clean_train_data_pos.txt', 'w')
    fo_n = open('./weibo_emotion/clean_train_data_neg.txt', 'w')

    weibo_pos = []
    weibo_neg = []
    jieba.load_userdict('./dict/dict.txt')
    reg_eng = re.compile(u'[a-zA-Z0-9@#]+')

    for line in fp.readlines():
        line = ''.join(line.strip().decode('utf-8').split())
        line = re.sub(reg_eng, '', line)
        line = jieba.lcut(line)
        if len(line) > 7:
            line = ' '.join(line)
            weibo_pos.append(line)
            print len(weibo_pos)
    print len(weibo_pos)

    for line in fn.readlines():
        line = ''.join(line.strip().decode('utf-8').split())
        line = re.sub(reg_eng, '', line)
        line = jieba.lcut(line)
        if len(line) > 7:
            line = ' '.join(line)
            weibo_neg.append(line)
            print len(weibo_neg)
    print len(weibo_neg)

    for line in weibo_pos:
        fo_p.write('%s\n' % line.encode('utf-8'))
    for line in weibo_neg:
        fo_n.write('%s\n' % line.encode('utf-8'))

    fp.close()
    fn.close()
    fo_p.close()
    fo_n.close()


def make_training_data():
    f = open('./weibo/data.txt', 'r')
    f_out = open('./weibo/train_set.txt', 'w')
    jieba.load_userdict('./dict/dict.txt')
    num = 0
    for line in f.readlines():
        line = line.strip().decode('utf-8').split('|**|')
        weibo = HanziConv.toSimplified(line[1])
        weibo = preprocess_weibo(weibo)
        seg_list = jieba.lcut(weibo)
        seg_list = [w for w in seg_list if w not in punc and w not in punc_en]
        if len(seg_list) > 7:
            weibo = ' '.join(seg_list)
            f_out.write('%s %d\n' % (weibo.encode('utf-8'), int(line[2])))
            num += 1
            print num
    f.close()
    f_out.close()


def make_testing_data():
    # fn = open('./NLPCC/clean_neg.txt', 'r')
    # fp = open('./NLPCC/clean_pos.txt', 'r')
    # ft = open('./NLPCC/clean_test.txt', 'r')
    # fc = open('./NLPCC/emotion_test_set.txt', 'r')
    f_nlp13 = open('./NLPCC/preprocess/test_set_nlpcc13.txt', 'r')
    # fo = open('./NLPCC/test_data_nlpcc14_review_1.txt', 'w')
    fo = open('./NLPCC/test_data_nlpcc13_weibo.txt', 'w')
    jieba.load_userdict('./dict/dict.txt')

    for line in f_nlp13.readlines():
        line = line.strip().decode('utf-8').split(' ')
        weibo = ''.join(line[:-1])
        weibo = HanziConv.toSimplified(weibo)
        weibo = preprocess_weibo(weibo)
        seg_list = jieba.lcut(weibo)
        seg_list = [w for w in seg_list if w not in punc and w not in punc_en and w not in [u'\u3000']]
        if seg_list:
            weibo = ' '.join(seg_list)
            fo.write('%s %d\n' % (weibo.encode('utf-8'), int(line[-1])))
    '''
    for line in fn.readlines():
        line = line.strip().decode('utf-8')
        weibo = HanziConv.toSimplified(line)
        weibo = preprocess_weibo(weibo)
        seg_list = jieba.lcut(weibo)
        seg_list = [w for w in seg_list if w not in punc and w not in punc_en and w not in [u'\u3000']]
        if seg_list:
            weibo = ' '.join(seg_list)
            fo.write('%s %d\n' % (weibo.encode('utf-8'), 0))
    '''

    # fp.close()
    f_nlp13.close()
    fo.close()


def process_txt():

    fn = open('./NLPCC/sample.negative.txt', 'r')
    fn_out = open('./NLPCC/clean_neg.txt', 'w')
    '''
    fp = open('./NLPCC/sample.positive.txt', 'r')
    fp_out = open('./NLPCC/clean_pos.txt', 'w')
    '''
    # ft = open('./NLPCC/test.label.cn.txt', 'r')
    # ft_out = open('./NLPCC/clean_test.txt', 'w')
    temp = []
    for line in fn.readlines():
        line = line.strip().decode('utf-8')
        if line:
            temp.append(line)
        if line == u'</review>':
            weibo = ' '.join(temp[1:-1])
            weibo = re.sub(u'\r', ' ', weibo)
            # label = int(re.findall(u'"[0-9]+"', temp[0])[1].replace('"', ''))
            fn_out.write('%s\n' % weibo.encode('utf-8'))
            temp = []
    fn.close()
    fn_out.close()
    '''
    fp.close()
    fp_out.close()
    ft.close()
    ft_out.close()
    '''


def process_xml():
    fo = open('./NLPCC/train_set_nlpcc13.txt', 'w')
    tree = ET.parse('./NLPCC/nlpcc13_train.xml')
    root = tree.getroot()
    weibo = []
    pos = ['happiness', 'like', 'surprise']
    neg = ['disgust', 'fear', 'anger', 'sadness']
    neu = ['none']
    for c in root:
        labels = c.attrib['emotion-type']
        if labels in pos:
            label = 2
        elif labels in neg:
            label = 0
        elif labels in neu:
            label = 1
        else:
            print "error: not find a label"
            return
        for s in c:
            if s.text:
                weibo.append(s.text)
        str_weibo = ' '.join(weibo)
        fo.write('%s %d\n' % (str_weibo.encode('utf-8'), label))
        weibo = []
    fo.close()


def main():
    '''
    dir_s = './2012_weibo/weibo%s.txt'
    dir_t = './weibo_emotion/week%s.txt'
    for i in range(1, 5):
        s = dir_s % str(i)
        t = dir_t % str(i+76)
        find_emotion(s, t)
    '''
    f = open('./NLPCC/preprocess/train_set_nlpcc14.txt', 'r')
    fo = open('./NLPCC/preprocess/train_set_nlpcc13.txt', 'w')

    for line in f.readlines()[:10000]:
        fo.write('%s' % line)

    f.close()
    fo.close()


def clean_weibo():
    f = open('./2012_weibo/weibo_freshdata.2016-10-07', 'r')
    dir = './2012_weibo/weibo%s.txt'
    j = 0
    weibo = []
    # reg = re.compile(u'<[\u4E00-\u9FFF\u2022A-Za-z0-9 .\'\-\[\]~《》·#=:_"/%?]+>')
    reg = re.compile(u'<[^<>]+[>|"]')
    for i in range(20000000):
        if i % 5000000 == 0:
            j += 1
            d = dir % str(j)
            fs = open(d, 'w')
        line = f.readline()
        line = line.strip().split('\t')[9].decode('utf-8')
        line = re.sub(reg, '', line)
        fs.write('%s\n' % line.encode('utf-8'))
    f.close()
    fs.close()


def select_data():
    fp = open('./weibo_emotion/weibo_pos.txt', 'w')
    fn = open('./weibo_emotion/weibo_neg.txt', 'w')
    dir = "./weibo_emotion/week"
    for i in range(1, 81):
        filename = dir + str(i) + '.txt'
        f = open(filename, 'r')
        for line in f.readlines():
            if 2 == int(line.split(',')[1]):
                fp.write('%s\n' % line.split(',')[0])
            else:
                fn.write('%s\n' % line.split(',')[0])
        f.close()
        print i
    fp.close()
    fn.close()


def get_train_data():
    fp = open('./weibo_emotion/weibo_pos.txt', 'r')
    fn = open('./weibo_emotion/weibo_neg.txt', 'r')
    ftp = open('./weibo_emotion/train_data_pos.txt', 'w')
    ftn = open('./weibo_emotion/train_data_neg.txt', 'w')
    weibo_pos = []
    weibo_neg = []
    k = 5000000
    for line in fp.readlines():
        weibo_pos.append(line)
    weibo_pos = random.sample(weibo_pos, k)
    train_data_pos = []
    for x in weibo_pos:
        ftp.write('%s' % x)
    for line in fn.readlines():
        weibo_neg.append(line)
    weibo_neg = random.sample(weibo_neg, k)
    train_data_neg = []
    for x in weibo_neg:
        ftn.write('%s' % x)
    fp.close()
    fn.close()
    ftp.close()
    ftn.close()


def create_custom_dict():
    f1 = open('./dict/SogouR.txt', 'r')
    f2 = open('./dict/dict_.txt', 'r')
    f3 = open('./dict/pos_words.txt', 'r')
    f4 = open('./dict/neg_words.txt', 'r')
    f5 = open('./dict/neu_words.txt', 'r')
    fd = open('./dict/dict.txt', 'w')

    words = []
    for line in f1.readlines():
        words.extend(line.strip().split()[0].decode('utf-8').split('-'))
        # words.append(line.strip().split()[0].split('-')[0].decode('utf-8'))
        # words.append(line.strip().split()[0].split('-')[1].decode('utf-8'))
    print len(words)
    for line in f2.readlines():
        words.append(line.strip().decode('utf-8'))
    print len(words)
    for line in f3.readlines():
        words.append(line.strip().decode('utf-8'))
    print len(words)
    for line in f4.readlines():
        words.append(line.strip().decode('utf-8'))
    print len(words)
    for line in f5.readlines():
        words.append(line.strip().decode('utf-8'))
    print len(words)

    words = set(words)
    print len(words)

    for word in words:
        fd.write("%s\n" % word.encode('utf-8'))

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    fd.close()


def create_chinese_dict():
    f1 = open(u'./dict/主张词语.txt', 'r')
    f2 = open(u'./dict/程度级别词语.txt', 'r')
    f = open('./dict/chinese_words.txt', 'r')
    fo = open('./dict/chinese_useless_words.txt', 'w')

    words = []
    for w in f1.readlines():
        words.append(w.strip().decode('utf-8'))
    for w in f2.readlines():
        words.append(w.strip().decode('utf-8'))
    for w in f.readlines():
        words.append(w.strip().decode('utf-8'))
    words = set(words)
    for w in words:
        fo.write('%s\n' % w.encode('utf-8'))
    f1.close()
    f2.close()
    f.close()


def get_emotion_word():
    fp1 = open('./dict/ntusd/ntusd-positive.txt', 'r')
    fp2 = open(u'./dict/正面情感词语.txt', 'r')
    fp3 = open(u'./dict/正面评价词语.txt', 'r')
    fn1 = open('./dict/ntusd/ntusd-negative.txt', 'r')
    fn2 = open(u'./dict/负面情感词语.txt', 'r')
    fn3 = open(u'./dict/负面评价词语.txt', 'r')

    fo_p = open('./dict/pos_words.txt', 'w')
    fo_n = open('./dict/neg_words.txt', 'w')

    pos = []
    neg = []
    for w in fp1.readlines():
        pos.append(w.strip())
    for w in fp2.readlines():
        pos.append(w.strip())
    for w in fp3.readlines():
        pos.append(w.strip())
    print len(pos)
    for w in pos:
        fo_p.write('%s\n' % w)

    for w in fn1.readlines():
        neg.append(w.strip())
    for w in fn2.readlines():
        neg.append(w.strip())
    for w in fn3.readlines():
        neg.append(w.strip())
    print len(neg)
    for w in neg:
        fo_n.write('%s\n' % w)

    fp1.close()
    fp2.close()
    fp3.close()
    fn1.close()
    fn2.close()
    fn3.close()
    fo_p.close()
    fo_n.close()

if __name__ == '__main__':
    # read_file()
    # preprocess_weibo('')
    # find_emotion()
    # create_custom_dict()
    # select_data()
    # main()
    # get_train_data()
    # clean_weibo()
    # create_chinese_dict()
    # get_emotion_word()
    # process_weibo_deep()
    # make_training_data()
    make_testing_data()
    # process_txt()
    # process_xml()
