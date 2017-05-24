import tensorflow as tf
import numpy as np
a = [-1, 3, 4, 1, 2, -1]
graph = tf.Graph()
with graph.as_default():
    '''
    embeddings = tf.Variable(
        tf.random_uniform([5, 5], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, a)
    t = tf.reshape(embed, [2, 5*3])
    sent_w = tf.Variable(
        tf.truncated_normal([2, 5],
                            stddev=1.0))
    sent_b = tf.Variable(tf.zeros([2]))
    sent_logits = tf.matmul(embed, sent_w, transpose_b=True) + sent_b
    sig = tf.sigmoid(sent_logits)
    reduce = tf.reduce_sum(sig)
    '''
    y = tf.placeholder(shape=[None], dtype=tf.int64)
    one_hot_y = tf.one_hot(y, 10)
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=one_hot_y)
    # sent_softmax = tf.nn.softmax(sent_logits)
    # max_index = tf.argmax(sent_softmax, axis=1)
    # one_hot_sent = tf.one_hot(max_index, 2, dtype=tf.int32)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    '''
    print embed.get_shape()
    print embeddings.eval()
    print embed.eval()
    print t.eval()
    '''
    sess.run(ce, {y: []})
    '''
    print sent_softmax.get_shape()
    print sent_softmax.eval()
    print max_index.eval()
    print one_hot_sent.eval()
    '''
