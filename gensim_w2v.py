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
    print len(tweets)
    dictionary = Dictionary(tweets)
    corpus = [dictionary.doc2bow(t) for t in tweets]

    lda = LdaMulticore(corpus=corpus, id2word=dictionary, workers=multiprocessing.cpu_count()-1, num_topics=20)
    lda.save('ldamodel.lda')
    # lda = LdaMulticore.load('ldamodel.lda')
    print lda.print_topics(5)

if __name__ == '__main__':
    model()
