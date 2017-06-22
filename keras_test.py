from keras.layers import Lambda
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np

x = np.random.randint(1, 5, (2, 2))
y = np.random.randint(1, 5, (2, 3))

a = tf.constant(x)
b = tf.constant(y)
a = tf.cast(a, tf.int32)
b = tf.cast(b, tf.int32)

var = K.dot(a, b)
sess = tf.Session()
print sess.run(a)
print sess.run(b)
print sess.run(var)
