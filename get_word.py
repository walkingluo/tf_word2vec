import sys
import os
import random
import nltk
import re
import time
import csv
import preprocessing

FULL_DATA = 'training.1600000.processed.noemoticon.csv'


def get_tweets(filename):
    fp = open(filename)
    rd = csv.reader(fp)

    tweets = []
    for row in rd:
        tweets.append([row[0]] + [row[5]])
    return tweets


def get_data(tweets, out):
    procTweets = [(preprocessing.processAll(text), sent) for (sent, text) in tweets]
    # print len(procTweets)
    # print procTweets[0]

    stemmer = nltk.stem.PorterStemmer()
    tweets_sent = []

    for (text, sent) in procTweets:
        words = [word if(word[0:2] == '__') else word.lower() for word in text.split()]
        words = [stemmer.stem(w) for w in words]
        if sent == '4':
            sent = '1'
        w = ' '.join(words)
        out.write("%s %s\n" % (str(w), sent))


def main():
    tweets = get_tweets('/home/jiangluo/sentiment_analysis/data/'+FULL_DATA)
    # print tweets[0]
    random.shuffle(tweets)
    print len(tweets)
    print tweets[:10]
    out = open('/home/jiangluo/tf_word2vec/word.txt', "w")
    get_data(tweets, out)
    out.close()
    print 'done'

if __name__ == "__main__":
    main()
