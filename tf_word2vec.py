# Tensorflow for word2vec
# coding=utf-8
from __future__ import division
# from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from test_data import main


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

'''
def read_tweet(filename):
    fp = open(filename)
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
tweets, tweets_sent, tweets_topic = read_tweet('/home/jiangluo/tf_word2vec/tweets.txt')


def set_words_sentiment(tweets, tweets_sent):
    words_sent = []
    for i in range(len(tweets)):
        words_sent.extend(len(tweets[i]) * [tweets_sent[i]])
    return words_sent
words_sent = set_words_sentiment(tweets, tweets_sent)


def set_words_topic(tweets, tweets_topic):
    words_topic = []
    for i in range(len(tweets)):
        words_topic.extend(len(tweets[i]) * [tweets_topic[i]])
    return words_topic
words_topic = set_words_topic(tweets, tweets_topic)


def get_word_lexicon():
    f = open('words_sent_lexicon.txt', 'r')
    words_sent_lexicon = []
    for s in f.readlines():
        words_sent_lexicon.append(int(s.strip()))
    return words_sent_lexicon
words_sent_lexicon = get_word_lexicon()


def tweets_to_wordlist(tweets):
    words = []
    for i in range(len(tweets)):
        words.extend(tweets[i])
    # print len(words)
    # print words[:10]
    return words

words = tweets_to_wordlist(tweets)
print('Data size: ', len(words))     # 22386665


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
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
'''
vocabulary_size, data, words_sent, vocab_counts, reverse_dictionary, neu_words, pos_words, neg_words = main()
print len(data)
print vocabulary_size
# vocabulary_size = 200000
data_index = 0
word_dict = dict()


# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    global word_dict
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels_sent = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    # labels_topic = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels_lexicon = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer_sent = collections.deque(maxlen=span)
    # buffer_topic = collections.deque(maxlen=span)
    # buffer_lexicon = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        buffer_sent.append(words_sent[data_index])
        # buffer_topic.append(words_topic[data_index])
        # buffer_lexicon.append(words_sent_lexicon[data_index])
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
            # labels_topic[i * num_skips + j, 0] = buffer_topic[skip_window]
            w = reverse_dictionary[buffer[skip_window]]
            try:
                temp = word_dict[w]
            except Exception, e:
                if w in neu_words:
                    temp = 2
                elif w in pos_words:
                    temp = 4
                elif w in neg_words:
                    temp = 0
                elif buffer_sent[skip_window] == 1:
                    temp = 3
                elif buffer_sent[skip_window] == 0:
                    temp = 1
                else:
                    print "error: not find in word dict!"
                    return
                word_dict[w] = temp
            labels_lexicon[i * num_skips + j, 0] = temp
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, labels_sent, labels_lexicon


def save_vec(filename, embeddings, reverse_dictionary):
    f = open(filename, 'w')
    f.write('%s %s\n' % (vocabulary_size, embedding_size))
    for i in range(vocabulary_size):
        word = reverse_dictionary[i]
        embed = ' '.join(list(embeddings[i].astype(np.str))).encode('utf-8')
        f.write("%s %s\n" % (word.encode('utf-8'), embed))
    f.close()
'''
batch, labels, labels_sent = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
'''
batch_size = 128
embedding_size = 100  # Dimension of the embedding vector.
skip_window = 5       # How many words to consider left and right.
num_skips = 8         # How many times to reuse an input to generate a label.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 128    # Number of negative examples to sample.

num_steps = int(len(data) / 16)
print num_steps

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('input'):
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        sent_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])
        # topic_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        lexicon_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.name_scope('global_step'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    with tf.name_scope('embeddings'):
        # Look up embeddings for inputs.
        init_width = 0.5 / embedding_size
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -init_width, init_width))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    with tf.name_scope('softmax'):
        # Construct the variables for the softmax loss
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(
            tf.zeros([vocabulary_size]))
    with tf.name_scope('sampled'):
        lables_matrix = tf.cast(train_labels, tf.int64)
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=lables_matrix,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=vocabulary_size,
            distortion=0.75,
            unigrams=vocab_counts)
    with tf.name_scope('true_logits'):
        labels = tf.reshape(train_labels, [batch_size])
        true_w = tf.nn.embedding_lookup(softmax_weights, labels)
        true_b = tf.nn.embedding_lookup(softmax_biases, labels)
        true_logits = tf.reduce_sum(tf.multiply(embed, true_w), 1) + true_b
    with tf.name_scope('sampled_logits'):
        sampled_w = tf.nn.embedding_lookup(softmax_weights, sampled_ids)
        sampled_b = tf.nn.embedding_lookup(softmax_biases, sampled_ids)
        sampled_b_vec = tf.reshape(sampled_b, [num_sampled])
        sampled_logits = tf.matmul(embed, sampled_w, transpose_b=True) + sampled_b_vec
    with tf.name_scope('loss_w'):
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        loss_w = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / batch_size

    '''
    loss_w = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                 biases=softmax_biases,
                                 inputs=embed,
                                 labels=train_labels,
                                 num_sampled=num_sampled,
                                 num_classes=vocabulary_size))
    '''
    with tf.name_scope('sentiment_loss'):
        # weights and biases for sentiment
        sent_w = tf.Variable(
            tf.truncated_normal([1, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        sent_b = tf.Variable(tf.zeros([1]))
        sent_logits = tf.matmul(embed, sent_w, transpose_b=True) + sent_b
        sent_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=sent_logits, labels=sent_labels))
    '''
    with tf.name_scope('topic_loss'):
        # weights and biases for topic
        topic_w = tf.Variable(
            tf.truncated_normal([20, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        topic_b = tf.Variable(tf.zeros([20]))
        topic_logits = tf.matmul(embed, topic_w, transpose_b=True) + topic_b
        topic_label = tf.one_hot(topic_labels, 20, dtype=tf.int32)
        topic_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=topic_logits, labels=topic_label))
    '''
    with tf.name_scope('lexicon_loss'):
        # weights and biases for lexicon
        lexicon_w = tf.Variable(
            tf.truncated_normal([5, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        lexicon_b = tf.Variable(tf.zeros([5]))
        lexicon_logits = tf.matmul(embed, lexicon_w, transpose_b=True) + lexicon_b
        lexicon_label = tf.one_hot(lexicon_labels, 5, dtype=tf.int32)
        lexicon_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=lexicon_logits, labels=lexicon_label))

    with tf.name_scope('loss'):
        alpha = 0.3
        loss = (1 - alpha * 2) * loss_w + alpha * sent_loss + alpha * lexicon_loss

    # loss = loss_w
    with tf.name_scope('optimizer'):
        # Construct the SGD optimizer using a learning rate of 0.1.
        learning_rate = 0.2
        lr = tf.train.exponential_decay(learning_rate, global_step, num_steps, 0.005)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
    with tf.name_scope('summary'):
        tf.summary.scalar("loss", loss)
        # tf.summary.histogram('loss', loss)
        all_summary = tf.summary.merge_all()
    # Add variable initializer.
    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    filename = 'vec_s_l_400m_re_seg.txt'
    init.run()
    saver = tf.train.Saver()
    print("Initialized")

    ckpt = tf.train.get_checkpoint_state('./checkpoints_3/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    init_step = session.run(global_step)
    data_index = init_step * 16
    writer = tf.summary.FileWriter('./improved_graph_3/lr' + str(0.2), session.graph)
    average_loss = 0
    min_loss = 100
    for step in xrange(init_step, num_steps):
        batch_inputs, batch_labels, batch_sent, batch_lexicon = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, sent_labels: batch_sent,
                     lexicon_labels: batch_lexicon}
        '''
        batch_inputs, batch_labels, batch_sent, batch_lexicon, batch_topic = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, sent_labels: batch_sent,
                     lexicon_labels: batch_lexicon, topic_labels: batch_topic}
        '''
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val, lr_1, summary = session.run([optimizer, loss, lr, all_summary], feed_dict=feed_dict)
        average_loss += loss_val
        writer.add_summary(summary, global_step=step+1)
        if (step + 1) % 2000 == 0:
            average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step+1, ": ", average_loss, ":", lr_1, ":", len(word_dict))
            saver.save(session, './checkpoints_3/skip-gram', step+1)
            if step > 1500000 and average_loss < min_loss:
                final_embeddings = normalized_embeddings.eval()
                print('saving vector')
                save_vec(filename, final_embeddings, reverse_dictionary)
                min_loss = average_loss
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

print(data_index)


print('saving final vector')
save_vec(filename='vec_s_l_400m_re_seg_final.txt', final_embeddings, reverse_dictionary)
print 'done'


def plot_with_labels(low_dim_embs, labels, filename='tsne_text8.png'):
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
    # plt.rcParams['font.family'] = ['monospace']   # 指定默认字体
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
