# coding=utf-8
import os
import random
import jieba
import re
import string
from hanziconv import HanziConv
import numpy as np
from collections import Counter

fuhao = [u' ', u'。', u'，', u'《', u'》', u'（', u'）', u'【', u'】', u'〈', u'〉', u'；',
         u':', u'：', u'“ ', u'”', u'‘', u'’', u'·', u'—', u'…', u'–', u'．', u'、', u'~',
         u'`', u'.', u'&', u'+', u'=', u'“', u',', u'[', u']', u'<']
punc = "｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
punc = punc.decode("utf-8")

emotoin_pos = [u'[挤眼]', u'[亲亲]', u'[太开心]', u'[哈哈]', u'[酷]', u'[来]', u'[good]', u'[haha]',
               u'[ok]', u'[拳头]', u'[赞]', u'[耶]', u'[微笑]', u'[色]', u'[可爱]', u'[嘻嘻]',
               u'[爱你]', u'[心]', u'[鼓掌]', u'[馋嘴]', u'[抱抱_旧]', u'[发红包]', u'[礼物]', u'[害羞]',
               u'[玫瑰]', u'[威武]', u'[得意地笑]', u'[奥特曼]', u'[太阳]', u'[围观]', u'[哆啦A梦微笑]',
               u'[飞个吻]', u'[抱抱]', u'[蛋糕]', u'[兔子]', u'[喵喵]', u'[笑哈哈]', u'[花心]', u'[偷笑]',
               u'[偷乐]', u'[推荐]', u'[音乐]', u'[羊年大吉]', u'[噢耶]', u'[微风]', u'[月亮]', u'[话筒]',
               u'[好喜欢]', u'[好棒]', u'[羞嗒嗒]', u'[给力]', u'[江南style]', u'[鲜花]', u'[好爱哦]']
emotoin_neg = [u'[生病]', u'[失望]', u'[黑线]', u'[吐]', u'[委屈]', u'[悲伤', u'[衰]', u'[愤怒]',
               u'[感冒]', u'[最差]', u'[NO]', u'[怒骂]', u'[困]', u'[哈欠]', u'[打脸]', u'[笑cry]',
               u'[汗]', u'[泪]', u'[晕]', u'[抓狂]', u'[怒]', u'[doge]', u'[蜡烛]', u'[弱]', u'[睡觉]',
               u'[崩溃]', u'[拜拜]', u'[打哈气]', u'[泪流满面]', u'[哼]', u'[草泥马]', u'[挖鼻]', u'[鄙视]',
               u'[阴险]', u'[可怜]', u'[最右]', u'[挖鼻屎]', u'[悲伤]', u'[疑问]', u'[思考]', u'[浮云]',
               u'[伤心]', u'[囧]', u'[呵呵]', u'[吃惊]', u'[抠鼻屎]']


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


def find_emotion():
    f = open('./data/weibo_train_data.txt', 'r')
    weibo = []
    for line in f.readlines():
        weibo.append(line.strip().split('\t')[6].decode('utf-8'))
    print len(weibo)
    weibo = np.array(weibo)
    emotion = []
    for w in weibo:
        emotion.append(re.findall("(\[.*?\])", w))
    # print len(emotion)
    # print emotion[35]
    idx = []
    for i, w in enumerate(emotion):
        if w != []:
            idx.append(i)
    print len(idx)
    # print idx
    emotion = np.array(emotion)
    emotion = emotion[idx]
    print len(emotion)
    emotion_dict = dict()
    for em in emotion:
        for ee in em:
            try:
                emotion_dict[ee] += 1
            except Exception, e:
                emotion_dict[ee] = 1
    print len(emotion_dict)
    del emotion_dict[u'[iso]']
    del emotion_dict[u'[转载]']
    del emotion_dict[u'[PDF]']
    del emotion_dict[u'[图]']
    del emotion_dict[u'[视频]']
    for k, v in emotion_dict.items():
        if k not in emotoin_pos and k not in emotoin_neg:
            if emotion_dict[k] <= 500:
                del emotion_dict[k]
    print len(emotion_dict)
    '''
    for k in emotion_dict.keys():
            print k, emotion_dict[k]
    '''
    print 'pos: ', len(emotoin_pos)
    print 'neg: ', len(emotoin_neg)
    '''
    newA = Counter(emotion_dict)
    for k, v in newA.most_common(len(emotion_dict)):
        print k, v
    '''
    weibo = weibo[idx]
    print len(weibo)
    sent = []
    for e in emotion:
        neg_num = 0
        pos_num = 0
        for i in e:
            if i in emotoin_pos:
                pos_num += 1
            if i in emotoin_neg:
                neg_num += 1
        if pos_num > neg_num:
            sent.append(2)
        elif pos_num < neg_num:
            sent.append(0)
        else:
            sent.append(1)
    print len(sent)
    print idx[:10]
    print sent[:10]

if __name__ == '__main__':
    # read_file()
    # preprocess_weibo('')
    find_emotion()
