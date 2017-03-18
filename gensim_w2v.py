# coding=utf-8
import gensim
import multiprocessing
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_tweet():
    fn = open('./weibo_emotion/train_data_neg.txt', 'r')
    fp = open('./weibo_emotion/train_data_pos.txt', 'r')
    tweets = []
    # tweets_sent = []
    for line in fn.readlines()[:2000000]:
        tweets.append(line.strip().decode('utf-8').split())
        # tweets_sent.append(int(line.split()[-1]))

    for line in fp.readlines()[:2000000]:
        tweets.append(line.strip().decode('utf-8').split())

    fn.close()
    fp.close()
    # print len(tweets)
    # print tweets[:10]
    return tweets  # , tweets_sent


def model():
    '''
    model_w = gensim.models.Word2Vec.load_word2vec_format('/home/jiangluo/tf_word2vec/vec_w.txt', binary=False)
    model_s = gensim.models.Word2Vec.load_word2vec_format('/home/jiangluo/tf_word2vec/vec.txt', binary=False)
    print "only words: ", model_w.similarity('good', 'bad')
    print "words and sentiment: ", model_s.similarity('good', 'bad')
    model_text8 = gensim.models.Word2Vec.load_word2vec_format('/home/jiangluo/tf_word2vec/vec_text8.txt', binary=False)
    print model_text8.similarity('good', 'bad')
    '''
    tweets = read_tweet()
    print len(tweets)
    # print ' '.join(tweets[1])
    # print tweets[1]

    dictionary = Dictionary(tweets)
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(t) for t in tweets]
    # print corpus
    del tweets
    # lda = LdaMulticore(corpus=corpus, id2word=dictionary, workers=multiprocessing.cpu_count()-1, num_topics=10, passes=1)
    # lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1, chunksize=10000, passes=2)
    # lda.save('lda_weibo_200.lda')
    lda = LdaModel.load('lda_weibo_200.lda')
    # print lda.print_topics(20, 5)
    '''
    tw = tweets[0]
    # tw = [dictionary.doc2bow(t) for t in tw]
    tw = dictionary.doc2bow(tw)
    print tw
    '''
    '''

    fp1 = open('word.txt', 'r')
    fp2 = open('tweets.txt', 'w')

    all_tweets = []
    for line in fp1.readlines():
        all_tweets.append(line.rstrip())
    print 'done1'
    topic_tweet = []
    for i in range(len(corpus)):
        sort_topics = list(sorted(lda[corpus[i]], key=lambda x: x[1]))
        topic_tweet.append(sort_topics[-1][0])
        print len(topic_tweet)
    print 'done2'
    for i in range(len(all_tweets)):
        fp2.write('%s %s\n' % (all_tweets[i], topic_tweet[i]))
    print 'done3'
    fp1.close()
    fp2.close()
    '''

    fn = open('./weibo_emotion/train_data_neg.txt', 'r')
    fp = open('./weibo_emotion/train_data_pos.txt', 'r')
    fo = open('./weibo_emotion/train_weibo_200M.txt', 'w')
    weibo = []
    for line in fn.readlines()[:2000000]:
        weibo.append(line.rstrip())
    for line in fp.readlines()[:2000000]:
        weibo.append(line.rstrip())
    print 'done1'
    topic = []
    for i in range(len(corpus)):
        sort_topics = list(sorted(lda[corpus[i]], key=lambda x: x[1]))
        topic.append(sort_topics[-1][0])
        print len(topic)
    print 'done2'
    for i in range(len(weibo)):
        fo.write('%s %s\n' % (weibo[i], topic[i]))
    print 'done3'
    fn.close()
    fp.close()
    fo.close()
    # print a[-1]
    # print lda.print_topic(a[-1][0])


def test():
    print multiprocessing.cpu_count()-1

if __name__ == '__main__':
    model()
    # test()
