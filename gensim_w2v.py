import gensim
import multiprocessing
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


def read_tweet(filename):
    fp = open(filename)
    tweets = []
    tweets_sent = []
    for line in fp.readlines():
        tweets.append(line.split()[:-1])
        # tweets_sent.append(int(line.split()[-1]))
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
    tweets = read_tweet('/home/jiangluo/tf_word2vec/word.txt')

    dictionary = Dictionary(tweets)
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(t) for t in tweets]
    # print corpus

    # lda = LdaMulticore(corpus=corpus, id2word=dictionary, workers=multiprocessing.cpu_count()-1, num_topics=2)
    # lda.save('lda.lda')
    lda = LdaMulticore.load('ldamodel.lda')
    '''
    tw = tweets[0]
    # tw = [dictionary.doc2bow(t) for t in tw]
    tw = dictionary.doc2bow(tw)
    print tw
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
    # print a[-1]
    # print lda.print_topic(a[-1][0])

if __name__ == '__main__':
    model()
