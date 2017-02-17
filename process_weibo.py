# coding=utf-8
import os
import random
import jieba


def read_file():
    fp1 = open('weibo/sample_0.1_1', 'r')   # “愤怒”
    fp2 = open('weibo/sample_0.1_2', 'r')   # “厌恶”
    fp3 = open('weibo/sample_0.1_3', 'r')   # “高兴”
    fp4 = open('weibo/sample_0.1_4', 'r')   # “低落”

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

    print negative_weibo[0].decode('gbk')
    print positive_weibo[0].decode('gbk')

    seg_list = jieba.lcut(negative_weibo[0])
    seg_list = [w for w in seg_list if w not in [u' ', u'。', u'，', u'《', u'》']]
    print '/'.join(seg_list)

if __name__ == '__main__':
    read_file()