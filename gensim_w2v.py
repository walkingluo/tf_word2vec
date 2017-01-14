import gensim


def model():
    model_w = gensim.models.Word2Vec.load_word2vec_format('/home/jiangluo/tf_word2vec/vec_w.txt', binary=False)
    model_s = gensim.models.Word2Vec.load_word2vec_format('/home/jiangluo/tf_word2vec/vec.txt', binary=False)
    print "only words: ", model_w.similarity('good', 'bad')
    print "words and sentiment: ", model_s.similarity('good', 'bad')

if __name__ == '__main__':
    model()
