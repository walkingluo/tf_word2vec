# coding=utf-8

from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dropout, Dense, Activation, SpatialDropout1D
from keras.layers import Embedding
from keras.callbacks import Callback
from keras import regularizers
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

'''
embeddings_index = {}
f = open('./glove.6B/glove.6B.100d.txt', 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
'''

'''
def load_train_data(filename):
    f = open(filename, 'r')
    train = []
    label = []
    for line in f.readlines()[:100000]:
        line = line.strip().decode('utf-8').split()
        train.append(line[:-1])
        t_label = int(line[-1])
        if t_label == 2:
            label.append(1.)
        else:
            label.append(0.)
    return train, label
'''


def load_train_test_data(filename):
    f = open(filename, 'r')
    test = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        t_label = int(line[-1])
        if t_label == 1:
            label.append(0)
        elif t_label == 2:
            label.append(1)
        else:
            label.append(2)
        test.append(line[:-1])
    return test, label

train, train_label = load_train_test_data('./NLPCC/train_data_nlpcc13_weibo.txt')
test, test_label = load_train_test_data('./NLPCC/test_data_nlpcc13_weibo.txt')

categorical_train_label = to_categorical(train_label, num_classes=3)
categorical_test_label = to_categorical(test_label, num_classes=3)


def read_vec(filename):
    f = open(filename)
    vocabulary_size, embedding_dim = f.readline().split()
    embeddings = []
    words = []
    for line in f.readlines():
        words.append(line.split()[0].decode('utf-8'))
        embeddings.append([float(num) for num in line.split()[1:]])
        # embeddings.append(line.split()[1:])
    f.close()
    return vocabulary_size, embedding_dim, embeddings, words

vocabulary_size, embedding_dim, embeddings, words = read_vec('vec_weibo_s_l.txt')
vocabulary_size = int(vocabulary_size)
embedding_dim = int(embedding_dim)
embeddings = np.array(embeddings)
print len(words)


def build_dict(words):
    words_dictionary = dict()
    for w in words:
        words_dictionary[w] = len(words_dictionary)
    return words_dictionary


def word_to_id(tweets, dictionary):
    tweets_to_id = []
    for tweet in tweets:
        tweet_to_id = []
        for w in tweet:
            try:
                tweet_to_id.append(dictionary[w])
            except Exception as e:
                tweet_to_id.append(0)
        tweets_to_id.append(tweet_to_id)
    return tweets_to_id

dictionary = build_dict(words)
train_id = word_to_id(train, dictionary)
test_id = word_to_id(test, dictionary)
# print len(train_id)

'''
count = 0
for i in range(vocabulary_size):
    try:
        embeddings[i] = embeddings_index[words[i]]
    except Exception, e:
        embeddings[i] = np.zeros(embedding_dim)
        count += 1
        continue
print count
'''

train_num = int(len(train_id) * 0.9)
# vaild_num = train_num + int(len(test_id) * 0.1)
# print train_num, vaild_num
max_weibo_length = 140

X_train = train_id[:train_num]
y_train = np.array(categorical_train_label[:train_num])
X_train = sequence.pad_sequences(X_train,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')

X_valid = train_id[train_num:]
y_vaild = np.array(categorical_train_label[train_num:])
X_valid = sequence.pad_sequences(X_valid,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')

X_test = test_id[:]
y_test = np.array(categorical_test_label[:])
X_test = sequence.pad_sequences(X_test,
                                maxlen=max_weibo_length,
                                padding='post',
                                truncating='post')
# print len(X_train)
# print len(X_test)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
'''
model.add(Embedding(vocabulary_size,
                    embedding_dim,
                    input_length=max_weibo_length))
'''
model.add(Embedding(vocabulary_size,
                    embedding_dim,
                    weights=[embeddings],
                    trainable=False,
                    input_length=max_weibo_length))

model.add(SpatialDropout1D(0.3))

model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

history = LossHistory()
model.fit(X_train, y_train, validation_data=(X_valid, y_vaild), epochs=5,
          batch_size=16, callbacks=[history])

score = model.evaluate(X_test, y_test)

y_p = model.predict_classes(X_test)
test_label = np.array(test_label)
print
print test_label[:20]
print y_p[:20]
acc = accuracy_score(test_label, y_p)
# weighted_f1 = f1_score(test_label, y_p, average='weighted')
macro_precision = precision_score(test_label, y_p, average='macro')
macro_recall = recall_score(test_label, y_p, average='macro')
macro_f1 = f1_score(test_label, y_p, average='macro')
micro_precision = precision_score(test_label, y_p, average='micro')
micro_recall = recall_score(test_label, y_p, average='micro')
micro_f1 = f1_score(test_label, y_p, average='micro')
print 'Acc: ', acc
# print 'Weighted F1: ', weighted_f1
print 'Macro precision: ', macro_precision
print 'Macro recall: ', macro_recall
print 'Macro F1: ', macro_f1
print 'Micro precision: ', micro_precision
print 'Micro recall: ', micro_recall
print 'Micro F1: ', micro_f1

print model.metrics_names
print score
print "Test loss: ", score[0]
print "Test accuracy: ", score[1]

'''
print history.history.keys()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
