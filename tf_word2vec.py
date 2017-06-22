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

vocabulary_size, weibo_id, weibo_sent, vocab_counts, reverse_dictionary, neu_words, pos_words, neg_words = main()
# vocabulary_size, weibo_id, vocab_counts, reverse_dictionary = main()

# print len(data)
# vocabulary_size = 200000
data_index = 0
max_length = 80
# word_dict = dict()


def get_batch(batch_size, skip_window):
    global data_index
    if data_index+batch_size > len(weibo_id):
        data_index = len(weibo_id)-batch_size
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
    if num_l == 0:
        num_l = 1
        word_l.append(0)
        label_d.append(1)
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
                batch_t[i, j] = 0
        label_t[i, 0] = weibo_s[i]
    return batch_w, label_w, batch_l, label_l, batch_t, label_t


# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # global word_dict
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    '''
    labels_sent = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    labels_lexicon = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    '''
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    # buffer_sent = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        # buffer_sent.append(words_sent[data_index])
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
            '''
            labels_sent[i * num_skips + j, 0] = float(buffer_sent[skip_window])
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
            '''
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels  # , labels_sent, labels_lexicon


def save_vec(filename, embeddings, reverse_dictionary):
    f = open(filename, 'w')
    f.write('%s %s\n' % (vocabulary_size, embedding_size))
    for i in range(vocabulary_size):
        word = reverse_dictionary[i]
        embed = ' '.join(list(embeddings[i].astype(np.str))).encode('utf-8')
        f.write("%s %s\n" % (word.encode('utf-8'), embed))
    f.close()

batch_size = 32
embedding_size = 100  # Dimension of the embedding vector.
skip_window = 10      # How many words to consider left and right.
num_skips = 16         # How many times to reuse an input to generate a label.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

num_steps = int(len(weibo_id) / batch_size) * 10
print num_steps

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('input'):
        # Input data.
        train_inputs_w = tf.placeholder(tf.int32)
        train_labels_w = tf.placeholder(tf.int32)
        train_inputs_l = tf.placeholder(tf.int32)
        train_labels_l = tf.placeholder(tf.int32)
        train_inputs_t = tf.placeholder(shape=[batch_size, max_length], dtype=tf.int32)
        train_labels_t = tf.placeholder(shape=[batch_size, 1], dtype=tf.int32)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.name_scope('global_step'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    with tf.name_scope('embeddings'):
        # Look up embeddings for inputs.
        # init_width = 0.5 / embedding_size
        init_width = 1.0
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -init_width, init_width))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs_w)
        embed_l = tf.nn.embedding_lookup(embeddings, train_inputs_l)
        input_t = tf.reshape(train_inputs_t, shape=[batch_size*max_length])
        embed_temp = tf.nn.embedding_lookup(embeddings, input_t)
        embed_t = tf.reshape(embed_temp, shape=[batch_size, embedding_size*max_length])
    with tf.name_scope('softmax'):
        # Construct the variables for the softmax loss
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(
            tf.zeros([vocabulary_size]))
    with tf.name_scope('sampled'):
        lables_matrix = tf.cast(train_labels_w, tf.int64)
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=lables_matrix,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=vocabulary_size,
            distortion=0.75,
            unigrams=vocab_counts)
    with tf.name_scope('true_logits'):
        labels = tf.reshape(train_labels_w, [-1])
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

        loss_w = (tf.reduce_mean(true_xent) + tf.reduce_mean(sampled_xent))
    with tf.name_scope('lexicon_loss'):
        # weights and biases for lexicon
        lexicon_w = tf.Variable(
            tf.truncated_normal([3, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        lexicon_b = tf.Variable(tf.zeros([3]))
        lexicon_logits = tf.matmul(embed_l, lexicon_w, transpose_b=True) + lexicon_b
        labels_l = tf.one_hot(train_labels_l, 3, dtype=tf.int32)
        lexicon_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=lexicon_logits, labels=labels_l))
    with tf.name_scope('sentiment_loss'):
        # weights and biases for sentiment
        sent_w = tf.Variable(
            tf.truncated_normal([3, embedding_size*max_length],
                                stddev=1.0 / math.sqrt(embedding_size)))
        sent_b = tf.Variable(tf.zeros([3]))
        sent_logits = tf.matmul(embed_t, sent_w, transpose_b=True) + sent_b
        labels_t = tf.one_hot(train_labels_t, 3, dtype=tf.int32)
        sent_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=sent_logits, labels=labels_t))
    with tf.name_scope('loss'):
        a = 0.5
        b = 0.8
        loss = b * (a * loss_w + (1 - a) * lexicon_loss) + (1 - b) * sent_loss
    with tf.name_scope('optimizer'):
        learning_rate = 0.25
        lr = tf.train.exponential_decay(learning_rate, global_step, num_steps, 0.004)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss_w, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
    '''
    with tf.name_scope('summary'):
        tf.summary.scalar("loss", loss)
        all_summary = tf.summary.merge_all()
    '''
    # Add variable initializer.
    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    # filename = 'wordvec_1.txt'
    init.run()
    saver = tf.train.Saver()
    print("Initialized")

    ckpt = tf.train.get_checkpoint_state('./checkpoints_m/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    init_step = session.run(global_step)
    data_index = init_step * batch_size
    # writer = tf.summary.FileWriter('./improved_graph/lr' + str(0.025), session.graph)
    average_loss = 0
    # min_loss = 100
    for step in xrange(init_step, num_steps):
        '''
        batch_inputs, batch_labels, batch_sent, batch_lexicon = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, sent_labels: batch_sent,
                     lexicon_labels: batch_lexicon}
        '''
        batch_w, label_w, batch_l, label_l, batch_t, label_t = get_batch(batch_size, skip_window)
        feed_dict = {train_inputs_w: batch_w, train_labels_w: label_w, train_inputs_l: batch_l, train_labels_l: label_l,
                     train_inputs_t: batch_t, train_labels_t: label_t}
        '''
        batch_w, label_w = get_batch(batch_size, skip_window)
        feed_dict = {train_inputs_w: batch_w, train_labels_w: label_w}
        '''
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val, lr_1 = session.run([optimizer, loss, lr], feed_dict=feed_dict)
        average_loss += loss_val
        # writer.add_summary(summary, global_step=step+1)
        if (step + 1) % 2000 == 0:
            average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step+1, ": ", average_loss, ":", lr_1)
            saver.save(session, './checkpoints_m/skip-gram', step+1)
            '''
            if step > 100000 and average_loss < min_loss:
                final_embeddings = normalized_embeddings.eval()
                print('saving vector')
                save_vec(filename, final_embeddings, reverse_dictionary)
                min_loss = average_loss
            '''
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
    final_embeddings = normalized_embeddings.eval()

print(data_index)


print('saving final vector')
filename_final = 'wordvec_2.txt'
save_vec(filename_final, final_embeddings, reverse_dictionary)
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
