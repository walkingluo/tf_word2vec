import gensim


def model():
    model = gensim.models.Word2Vec.load_word2vec_format('/home/jiangluo/tf_word2vec/vec.txt', binary=False)
    print model.similarity('good', 'bad')

if __name__ == '__main__':
    model()
