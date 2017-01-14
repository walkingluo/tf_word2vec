# Tensorflow for word2vec

from __future__ import division
# from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf


# Read the data into a list of strings.
'''
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data('text8.zip')
print('Data size', len(words))
'''


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
tweets, tweets_sent = read_tweet('/home/jiangluo/tf_word2vec/word.txt')


def set_words_sentiment(tweets, tweets_sent):
    words_sent = []
    for i in range(len(tweets)):
        words_sent.extend(len(tweets[i]) * [tweets_sent[i]])
    return words_sent
words_sent = set_words_sentiment(tweets, tweets_sent)


def tweets_to_wordlist(tweets):
    words = []
    for i in range(len(tweets)):
        words.extend(tweets[i])
    # print len(words)
    # print words[:10]
    return words

words = tweets_to_wordlist(tweets)
print 'Data size: ', len(words)     # 22386665
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

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
# print len(data)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels_sent = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
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
            labels_sent[i * num_skips + j, 0] = float(buffer_sent[skip_window])
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, labels_sent
'''
batch, labels, labels_sent = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
'''
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    sent_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the softmax loss
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss_w = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed, train_labels,
                                 num_sampled, vocabulary_size))

    # weights and biases for sentiment
    sent_w = tf.Variable(
        tf.truncated_normal([1, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    sent_b = tf.Variable(tf.zeros([1]))
    sent_logits = tf.matmul(embed, sent_w, transpose_b=True) + sent_b
    sent_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(sent_logits, sent_labels))
    alpha = 0
    loss = (1 - alpha) * loss_w + alpha * sent_loss
    # Construct the SGD optimizer using a learning rate of 0.1.
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 330001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels, batch_sent = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, sent_labels: batch_sent}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %r," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()

print data_index


def save_vec(embeddings, reverse_dictionary):
    f = open('vec_w.txt', 'w')
    f.write('%s %s\n' % (vocabulary_size, embedding_size))
    for i in range(vocabulary_size):
        word = reverse_dictionary[i]
        embed = ' '.join(list(embeddings[i].astype(np.str))).encode('utf-8')
        f.write("%s %s\n" % (word, embed))
    f.close()

print 'saving vector'
save_vec(final_embeddings, reverse_dictionary)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
