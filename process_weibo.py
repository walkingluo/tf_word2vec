# coding=utf-8
import os
import random
import jieba
import re
# import string
from hanziconv import HanziConv
import numpy as np
from collections import Counter

random.seed(1377)

punc = "。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
punc = punc.decode("utf-8")
punc_en = " \"$#%&'()*+,-./:;<=>[\]^_`{|}~·...._"
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

    riyu_re = re.compile(u'[\u3040-\u3210\u2000-\u2e40\uA000-\uFFFF\U00010000-\U0001ffff._]+')
    text_re = re.sub(riyu_re, '', text_re)

    handles_re = re.compile(u'#[\u4E00-\u9FFFA-Za-z0-9]+#')
    text_re = re.sub(handles_re, '_HANDLES_ ', text_re)

    url_re = re.compile(r'(http|https|ftp)://[a-zA-Z0-9\./]+')
    text_re = re.sub(url_re, '_URL_ ', text_re)

    repeat_re = re.compile(r'([!?])\1{1,}', re.IGNORECASE)

    def rpt_repl(match):
        return match.group(1)
    text_re = re.sub(repeat_re, rpt_repl, text_re)
    return text_re


def find_emotion():
    # f = open('./data/weibo_train_data.txt', 'r')
    f = open('./2012_weibo/week1.csv', 'r')
    weibo = []
    f.readline()
    for line in f.readlines():
        try:
            line = HanziConv.toSimplified(line.strip().split(',')[6].decode('utf-8'))
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
    print len(emotion)
    # print emotion[:10]

    idx = []
    for i, w in enumerate(emotion):
        if w != []:
            idx.append(i)
    print len(idx)
    emotion_clean = []
    for i in idx:
        emotion_clean.append(emotion[i])
    print len(emotion_clean)
    # print emotion_clean[:10]

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

    '''
    newA = Counter(emotion_dict)
    for k, v in newA.most_common(100):
        print k, v
    '''

    weibo_em = []
    for i in idx:
        weibo_em.append(weibo[i])
    print len(weibo_em)

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

    jieba.load_userdict('./dict/dict.txt')
    fw = open('./weibo_emotion/week1.txt', 'w')
    for i in range(len(sent)):
        if sent[i] != 1:
            # print weibo_em[i]
            weibo_r = preprocess_weibo(weibo_em[i])
            # fw.write('%s\n' % weibo_r.encode('utf-8'))
            seg_list = jieba.lcut(weibo_r)
            seg_list = [w for w in seg_list if w not in punc and w not in punc_en]
            if len(seg_list) > 5:
                weibo = ' '.join(seg_list)
                fw.write('%s,%d\n' % (weibo.encode('utf-8'), sent[i]))
    fw.close()


def create_custom_dict():
    f1 = open('./dict/lexicon_raw.txt', 'r')
    f2 = open('./dict/cyberword.txt', 'r')
    fd = open('./dict/dict.txt', 'w')

    words = []
    for word in f1.readlines():
        words.append(word.rstrip())
    print len(words)
    for line in f2.readlines():
        words.append(line.rstrip().split(',')[0])
    print len(words)
    f1.close()
    f2.close()

    for word in words:
        fd.write("%s\n" % word)
    fd.close()

if __name__ == '__main__':
    # read_file()
    # preprocess_weibo('')
    find_emotion()
    # create_custom_dict()
