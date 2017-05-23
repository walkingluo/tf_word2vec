import tensorflow as tf
import numpy as np
a = [1, 3, 4, 2, 3, 4]
graph = tf.Graph()
with graph.as_default():
    embeddings = tf.Variable(
        tf.random_uniform([5, 5], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, a)
    input_t = tf.reshape(embed, [2, 15])

    sent_w = tf.Variable(
        tf.truncated_normal([2, 5],
                            stddev=1.0))
    sent_b = tf.Variable(tf.zeros([2]))
    sent_logits = tf.matmul(embed, sent_w, transpose_b=True) + sent_b
    sig = tf.sigmoid(sent_logits)
    reduce = tf.reduce_sum(sig)
    # sent_softmax = tf.nn.softmax(sent_logits)
    # max_index = tf.argmax(sent_softmax, axis=1)
    # one_hot_sent = tf.one_hot(max_index, 2, dtype=tf.int32)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print embed.get_shape()
    print embeddings.eval()
    print embed.eval()
    print input_t.eval()
    '''
    print sent_softmax.get_shape()
    print sent_softmax.eval()
    print max_index.eval()
    print one_hot_sent.eval()
    '''
