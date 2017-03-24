# from __future__ import print_function
import os
import collections
import numpy as np
import random


def read_tweet(filename):
    fp = open(filename, 'rb')
    tweets = []
    tweets_sent = []
    tweets_topic = []
    for line in fp.readlines():
        tweets.append(line.split()[:-2])
        tweets_sent.append(int(line.split()[-2]))
        tweets_topic.append(int(line.split()[-1]))
    # print len(tweets)
    # print tweets[:10]
    return tweets, tweets_sent, tweets_topic


def read_weibo():
    fp = open('./weibo_emotion/clean_train_data_pos.txt', 'rb')
    fn = open('./weibo_emotion/clean_train_data_neg.txt', 'rb')
    weibo_pos = []
    weibo_neg = []
    for line in fp.readlines()[:3500000]:
        weibo_pos.append(line.decode('utf-8').split())
    for line in fn.readlines()[:3500000]:
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

vocabulary_size = 200000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
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


def main():
    # tweets, tweets_sent, tweets_topic = read_tweet('/home/jiangluo/tf_word2vec/tweets.txt')
    # tweets = tweets[:10000]
    # tweets_sent = tweets_sent[:10000]
    # print(len(tweets))
    # print tweets[0], tweets_sent[0], tweets_topic[0]
    num = 3500000
    weibo_pos, weibo_neg = read_weibo()
    # weibo_pos = random.sample(weibo_pos, len(weibo_neg))
    print len(weibo_pos), len(weibo_neg)
    print " ".join(weibo_pos[0])
    print " ".join(weibo_neg[0])

    weibo = []
    weibo.extend(weibo_pos)
    weibo.extend(weibo_neg)
    print len(weibo)
    # random.shuffle(weibo)
    del weibo_pos
    del weibo_neg
    index = np.arange(len(weibo))
    np.random.shuffle(index)
    weibo = np.array(weibo)
    weibo = weibo[index]

    weibo_sent = num * [1]
    weibo_sent.extend(num * [0])
    print len(weibo_sent)
    weibo_sent = np.array(weibo_sent)
    weibo_sent = weibo_sent[index]

    # print weibo_sent[:10], weibo_sent[-10:-1]
    ts = set_words_sentiment(weibo, weibo_sent)
    print len(ts)
    # tp = set_words_topic(tweets, tweets_topic)
    # print len(tp)
    words = tweets_to_wordlist(weibo)
    print(len(words))
    # print " ".join(words[:10])
    # neg_words, pos_words = read_lexicon()
    # set_words_sentment_in_lexicon(words, neg_words, pos_words)
    # words_sent_lexicon = get_word_lexicon()
    # print len(words_sent_lexicon)
    # print words_sent_lexicon[:10]
    del weibo
    del weibo_sent
    del index
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    vocab_counts = []
    for _, n in count:
        vocab_counts.append(n)
    # print vocab_counts[:10]
    del words
    del dictionary
    del count
    '''
    words_dict = dict()
    for w, n in count:
        words_dict[w] = n
    '''
    f_neu = open('./dict/neu_words.txt', 'r')
    neu_words = []
    for w in f_neu.readlines():
        neu_words.append(w.strip().decode('utf-8'))
    print len(neu_words)
    f_neu.close()
    '''
    count_num = 0
    for w in cn_words:
        try:
            count_num += words_dict[w]
        except Exception, e:
            continue
    print count_num
    '''
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
    return data, ts, vocab_counts, reverse_dictionary, neu_words, pos_words, neg_words

if __name__ == "__main__":
    main()
