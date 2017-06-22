# from __future__ import print_function
import os
import collections
import numpy as np
import random


def read_tweet(filename):
    fp = open(filename, 'rb')
    tweets = []
    tweets_sent = []
    for line in fp.readlines()[:100000]:
        if len(line.decode('utf-8').split()[:-1]) != 0:
            tweets.append(line.decode('utf-8').split()[:-1])
            tweets_sent.append(int(line.split()[-1]))
    return tweets, tweets_sent


def read_weibo():
    fp = open('./weibo_emotion/clean_train_data_pos.txt', 'rb')
    fn = open('./weibo_emotion/clean_train_data_neg.txt', 'rb')
    weibo_pos = []
    weibo_neg = []
    for line in fp.readlines()[:100000]:
        weibo_pos.append(line.decode('utf-8').split())
    for line in fn.readlines()[:100000]:
        weibo_neg.append(line.decode('utf-8').split())
    fp.close()
    fn.close()
    return weibo_pos, weibo_neg


def set_words_sentiment(tweets, tweets_sent):
    words_sent = []
    for i in range(len(tweets)):
        words_sent.extend(len(tweets[i]) * [tweets_sent[i]])
    return words_sent


def set_words_topic(tweets, tweets_topic):
    words_topic = []
    for i in range(len(tweets)):
        words_topic.extend(len(tweets[i]) * [tweets_topic[i]])
    return words_topic


def tweets_to_wordlist(tweets):
    words = []
    for i in range(len(tweets)):
        words.extend(tweets[i])
    # print len(words)
    # print words[:10]
    return words


def read_lexicon():
    fn = open('./opinion-lexicon-English/negative-words.txt', 'r')
    fp = open('./opinion-lexicon-English/positive-words.txt', 'r')
    neg_words = []
    pos_words = []
    for w in fn.readlines()[35:]:
        neg_words.append(w.strip())
    for w in fp.readlines()[35:]:
        pos_words.append(w.strip())
    fn.close()
    fp.close()
    print len(neg_words)
    print len(pos_words)
    # print pos_words
    return neg_words, pos_words


def set_words_sentment_in_lexicon(words, neg_words, pos_words):
    # 0 for neg
    # 1 for neu
    # 2 for pos
    f = open('words_sent_lexicon.txt', 'w')
    count = 0
    for w in words:
        if w in neg_words:
            f.write('%d\n' % 0)
        elif w in pos_words:
            f.write('%d\n' % 2)
        else:
            f.write('%d\n' % 1)
        count += 1
        print count
    f.close()


def get_word_lexicon():
    f = open('words_sent_lexicon.txt', 'r')
    words_sent_lexicon = []
    for s in f.readlines():
        words_sent_lexicon.append(int(s.strip()))
    return words_sent_lexicon

# vocabulary_size = 200000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, n in count:
        if n == -1 or n >= 5:
            dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0


# Function to generate a training batch for the skip-gram model.
def generate_batch_tweet(data, words_sent, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels_sent = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # labels_topic = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer_sent = collections.deque(maxlen=span)
    # buffer_topic = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        buffer_sent.append(words_sent[data_index])
        # buffer_topic.append(words_topic[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            labels_sent[i * num_skips + j, 0] = buffer_sent[skip_window]
            # labels_topic[i * num_skips + j, 0] = buffer_topic[skip_window]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, labels_sent  # , labels_topic


def load_data():
    f = open('./weibo/train_set.txt', 'r')
    train = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        train.append(line[:-1])
        t_label = int(line[-1])
        if t_label == 2:
            label.append(1)
        else:
            label.append(0)
    return train, label


def load_test_data(filename):
    f = open(filename, 'r')
    f_s = open('./subjective_weibo.txt', 'r')
    subjective_w = []
    test = []
    label = []
    num = 0
    none_num = 0
    for line in f_s.readlines():
        subjective_w.append(int(line.strip()))
    for line in f.readlines():
        if num in subjective_w:
            line = line.strip().decode('utf-8').split()
            if int(line[-1]) == 0:
                none_num += 1
            label.append(int(line[-1]) - 1)
            test.append(line[:-1])
        num += 1
    return test, label, none_num

max_length = 80


def get_batch(weibo_id, weibo_sent, batch_size, skip_window, reverse_dictionary, neu_words, pos_words, neg_words):
    global data_index
    weibo = weibo_id[data_index:data_index+batch_size]
    weibo_s = weibo_sent[data_index:data_index+batch_size]
    data_index = (data_index+batch_size) % len(weibo_id)
    size = 0
    word_l = []
    label_d = []
    num_l = 0
    for i in range(batch_size):
        weibo_len = len(weibo[i])
        length = 0
        begin = 1
        if skip_window+1-weibo_len > 1:
            begin = skip_window+1-weibo_len
        for l in range(begin, skip_window+1):
            length += l
        size += weibo_len * skip_window * 2 - 2 * length
        for w in weibo[i]:
            if reverse_dictionary[w] in neu_words:
                word_l.append(w)
                label_d.append(1)
                num_l += 1
            elif reverse_dictionary[w] in neg_words:
                word_l.append(w)
                label_d.append(0)
                num_l += 1
            elif reverse_dictionary[w] in pos_words:
                word_l.append(w)
                label_d.append(2)
                num_l += 1
            else:
                continue
    batch_w = np.ndarray(shape=(size), dtype=np.int32)
    label_w = np.ndarray(shape=(size, 1), dtype=np.int32)
    batch_l = np.ndarray(shape=(num_l), dtype=np.int32)
    label_l = np.ndarray(shape=(num_l, 1), dtype=np.int32)
    batch_t = np.ndarray(shape=(batch_size, max_length), dtype=np.int32)
    label_t = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    num = 0
    for i in range(batch_size):
        weibo_len = len(weibo[i])
        for j in range(weibo_len):
            for z in range(1, skip_window+1):
                if j-z >= 0:
                    batch_w[num] = weibo[i][j]
                    label_w[num, 0] = weibo[i][j-z]
                    num += 1
                if j+z < weibo_len:
                    batch_w[num] = weibo[i][j]
                    label_w[num, 0] = weibo[i][j+z]
                    num += 1
    for i in range(num_l):
        batch_l[i] = word_l[i]
        label_l[i] = label_d[i]
    for i in range(batch_size):
        weibo_len = len(weibo[i])
        for j in range(max_length):
            if j < weibo_len:
                batch_t[i, j] = weibo[i][j]
            else:
                batch_t[i, j] = -1
        label_t[i, 0] = weibo_s[i]
    return batch_w, label_w, batch_l, label_l, batch_t, label_t


def main():
    weibo, weibo_sent = read_tweet('./weibo_emotion/week1_s.txt')

    print len(weibo), len(weibo_sent)
    print weibo[0], weibo_sent[0]
    # print tweets[0], tweets_sent[0], tweets_topic[0]
    for wei in weibo:
        if len(wei) == 0:
            print 'error'
    '''
    num = 100000
    weibo_pos, weibo_neg = read_weibo()
    # weibo_pos = random.sample(weibo_pos, len(weibo_neg))
    print len(weibo_pos), len(weibo_neg)
    print " ".join(weibo_pos[0])
    print " ".join(weibo_neg[0])

    weibo = []
    weibo.extend(weibo_pos)
    weibo.extend(weibo_neg)
    print len(weibo)
    '''
    '''
    f_top = open('./weibo_emotion/train_weibo_200M.txt', 'r')
    weibo = []
    top = []
    for line in f_top.readlines():
        line = line.strip().decode('utf-8').split()
        weibo.append(line[:-1])
        top.append(int(line[-1]))
    print len(weibo)
    print len(top)
    '''
    # random.shuffle(weibo)
    # del weibo_pos
    # del weibo_neg
    '''
    index = np.arange(len(weibo))
    np.random.shuffle(index)
    weibo = np.array(weibo)
    weibo = weibo[index]

    weibo_sent = num * [0]
    weibo_sent.extend(num * [1])
    print len(weibo_sent)

    weibo_sent = np.array(weibo_sent)
    weibo_sent = weibo_sent[index]
    # print weibo_sent[:10], weibo_sent[-10:-1]
    '''
    '''
    ts = set_words_sentiment(weibo, weibo_sent)
    print len(ts)
    '''
    words = tweets_to_wordlist(weibo)
    print(len(words))
    # print " ".join(words[:10])
    # neg_words, pos_words = read_lexicon()
    # set_words_sentment_in_lexicon(words, neg_words, pos_words)
    # words_sent_lexicon = get_word_lexicon()
    # print len(words_sent_lexicon)
    # print words_sent_lexicon[:10]4
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print 'data len: ', len(data)
    print 'count len: ', len(count)
    print 'dictionary len: ', len(dictionary), reverse_dictionary[0]
    vocabulary_size = len(dictionary)
    word_id = []
    weibo_id = []
    for i in weibo:
        for w in i:
            try:
                word_id.append(dictionary[w])
            except Exception, e:
                word_id.append(0)
        weibo_id.append(word_id)
        word_id = []

    vocab_counts = []
    for _, n in count[:len(dictionary)]:
        vocab_counts.append(n)
    # print vocab_counts[:10]
    del words
    del dictionary
    del count
    f_neu = open('./dict/neu_words.txt', 'r')
    neu_words = []
    for w in f_neu.readlines():
        neu_words.append(w.strip().decode('utf-8'))
    print len(neu_words)
    f_neu.close()
    fp = open('./dict/pos_words.txt', 'r')
    fn = open('./dict/neg_words.txt', 'r')
    pos_words = []
    neg_words = []
    for w in fp.readlines():
        pos_words.append(w.strip().decode('utf-8'))
    for w in fn.readlines():
        neg_words.append(w.strip().decode('utf-8'))
    fp.close()
    fn.close()
    print len(pos_words), len(neg_words)
    '''
    batch, label, batch_l, label_l, batch_t, label_t = get_batch(weibo_id, weibo_sent, batch_size=16, skip_window=10,
                                                                 reverse_dictionary=reverse_dictionary,
                                                                 neu_words=neu_words, pos_words=pos_words, neg_words=neg_words)
    print batch[:10], label[:10]
    print batch_l[:10], label_l[:10]
    print batch_t[0][:], label_t[0]
    '''
    # print len(dictionary)
    # print(len(data))   # 22386665
    # print('Most common words (+UNK)', count[:10])
    # print('Sample data', data[:18], [reverse_dictionary[i] for i in data[:18]])
    '''
    batch, labels, labels_sent = generate_batch_tweet(data, ts, batch_size=128, num_skips=2, skip_window=1)
    # print " ".join(words[:8])
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]],
              labels_sent[i, 0])
    '''
    # print data_index
    # print batch
    # print labels.transpose()
    # return tweets, tweets_sent, dictionary, reverse_dictionary
    return vocabulary_size, weibo_id, weibo_sent, vocab_counts, reverse_dictionary, neu_words, pos_words, neg_words
    # return vocabulary_size, weibo_id, vocab_counts, reverse_dictionary


def save_train_data():
    weibo = read_tweet('./weibo_emotion/week1_train.txt')
    print len(weibo)
    f = open('./weibo_emotion/week1_train_1.txt', 'w')
    for w in weibo[:100000]:
        line = ' '.join(w)
        f.write('%s\n' % line.encode('utf-8'))
    f.close()


def read_vec(filename):
    f = open(filename)
    vocabulary_size, embedding_dim = f.readline().split()
    embeddings = []
    words = []
    emo = []
    emo_embed = []
    for line in f.readlines():
        w = line.split()[0].decode('utf-8')
        if w[0] == u'[':
            emo.append(w)
            emo_embed.append([float(num) for num in line.split()[1:]])
        words.append(line.split()[0].decode('utf-8'))
        embeddings.append([float(num) for num in line.split()[1:]])
        # embeddings.append(line.split()[1:])
    f.close()
    return vocabulary_size, embedding_dim, embeddings, words, emo, emo_embed


def load_user_weibo(filename):
    f = open(filename, 'r')
    train = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        if len(line) > 1:
            line = ' '.join(line)
            weibo = line.split(',')
        for w in weibo:
            train.append(w[:-1].split())
            label.append(int(w[-1]))
    return train, label

if __name__ == "__main__":
    # main()
    # save_train_data()
    # get_batch()
    # load_data()
    # load_test_data('./NLPCC/train_data_nlpcc14_weibo.txt')
    # test, label, none_num = load_test_data('./NLPCC/1/test_data_nlpcc13_weibo_new.txt')
    # print len(test), len(label), none_num
    '''
    vocabulary_size, embedding_dim, embeddings, words, emo, emo_embed = read_vec('vec_1.txt')
    print vocabulary_size
    print embedding_dim
    print len(words)
    print len(emo)
    print emo[0], emo[1], emo[2], emo[-1]
    '''
    train, label = load_user_weibo('./weibo_emotion/l_weibo.txt')
    print len(train), len(label)
    print train[:5]
    print label[:5]
