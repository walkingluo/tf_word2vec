import os
import collections
import numpy as np
import random


def read_tweet(filename):
    fp = open(filename)
    tweets = []
    tweets_sent = []
    for line in fp.readlines():
        tweets.append(line.split()[:-1])
        tweets_sent.append(int(line.split()[-1]))
    # print len(tweets)
    # print tweets[:10]
    return tweets, tweets_sent


def set_words_sentiment(tweets, tweets_sent):
    words_sent = []
    for i in range(len(tweets)):
        words_sent.extend(len(tweets[i]) * [tweets_sent[i]])
    return words_sent


def tweets_to_wordlist(tweets):
    words = []
    for i in range(len(tweets)):
        words.extend(tweets[i])
    # print len(words)
    # print words[:10]
    return words

vocabulary_size = 50000


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
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer_sent = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        buffer_sent.append(words_sent[data_index])
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
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, labels_sent


def main():
    tweets, tweets_sent = read_tweet('/home/jiangluo/tf_word2vec/word.txt')
    tweets = tweets[:300000]
    tweets_sent = tweets_sent[:300000]
    print len(tweets)
    ts = set_words_sentiment(tweets, tweets_sent)
    # print len(ts)
    words = tweets_to_wordlist(tweets)
    print len(words)
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print len(data)     # 22386665
    # print 'Most common words (+UNK)', count[:10]
    # print 'Sample data', data[:18], [reverse_dictionary[i] for i in data[:18]]
    setp = 60000
    for x in xrange(setp):
        batch, labels, labels_sent = generate_batch_tweet(data, ts, batch_size=128, num_skips=2, skip_window=1)
    print data_index
    # print batch
    # print labels.transpose()


if __name__ == "__main__":
    main()
