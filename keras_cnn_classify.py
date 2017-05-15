# coding=utf-8

from __future__ import division
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation, SpatialDropout1D, Merge, Reshape
from keras.layers import Embedding
from keras.callbacks import Callback
from keras import regularizers
from keras.models import load_model
import keras.backend as K
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
np.random.seed(1337)

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


def load_pos_neg_data(filename):
    f = open(filename, 'r')
    train = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        if int(line[-1]) in [4, 5, 6, 7]:
            train.append(line[:-1])
            label.append(0)
        elif int(line[-1]) in [1, 2]:
            train.append(line[:-1])
            label.append(1)
    return train, label


def load_4_classify_data(filename):
    f = open(filename, 'r')
    train = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        if int(line[-1]) == 1:
            train.append(line[:-1])
            label.append(0)
        elif int(line[-1]) == 2:
            train.append(line[:-1])
            label.append(1)
        elif int(line[-1]) == 7:
            train.append(line[:-1])
            label.append(2)
        elif int(line[-1]) == 4:
            train.append(line[:-1])
            label.append(3)
    return train, label


def subjective_classify(filename):
    f = open(filename, 'r')
    train = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        if int(line[-1]) == 0:
            train.append(line[:-1])
            label.append(0)
        else:
            train.append(line[:-1])
            label.append(1)
    return train, label


def load_train_data(filename):
    f = open(filename, 'r')
    test = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        if int(line[-1]) != 0:
            label.append(int(line[-1]) - 1)
            test.append(line[:-1])
    return test, label


def load_test_data(filename):
    f = open(filename, 'r')
    f_s = open('./subjective_weibo.txt', 'r')
    subjective_w = []
    test = []
    label = []
    num = 0
    for line in f_s.readlines():
        subjective_w.append(int(line.strip()))
    for line in f.readlines():
        if num in subjective_w:
            line = line.strip().decode('utf-8').split()
            label.append((int(line[-1]) - 1 + 8) % 8)
            test.append(line[:-1])
        num += 1
    return test, label


train, train_label = subjective_classify('./NLPCC/1/train_data_nlpcc13_weibo_new.txt')
test, test_label = subjective_classify('./NLPCC/1/test_data_nlpcc13_weibo_new.txt')
# train, train_label = load_train_data('./NLPCC/1/train_data_nlpcc13_weibo_new.txt')
# test, test_label = load_test_data('./NLPCC/1/test_data_nlpcc13_weibo_new.txt')
# train, train_label = load_pos_neg_data('./NLPCC/train_data_nlpcc13_weibo.txt')
# test, test_label = load_pos_neg_data('./NLPCC/test_data_nlpcc13_weibo.txt')
# train, train_label = load_4_classify_data('./NLPCC/train_data_nlpcc13_weibo.txt')
# test, test_label = load_4_classify_data('./NLPCC/test_data_nlpcc13_weibo.txt')

categorical_train_label = to_categorical(train_label, num_classes=2)
categorical_test_label = to_categorical(test_label, num_classes=2)


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

vocabulary_size, embedding_dim, embeddings, words = read_vec('vec_weibo_s_l_700m.txt')
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

train_num = int(len(train_id) * 0.95)
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

embedding_layer = Embedding(vocabulary_size,
                            embedding_dim,
                            weights=[embeddings],
                            trainable=False,
                            input_length=max_weibo_length)
embedding_layer_1 = Embedding(vocabulary_size,
                              embedding_dim,
                              weights=[embeddings],
                              trainable=True,
                              input_length=max_weibo_length)

model_1 = Sequential()
'''
model.add(Embedding(vocabulary_size,
                    embedding_dim,
                    input_length=max_weibo_length))
'''
model_1.add(embedding_layer)
'''
model_1.add(SpatialDropout1D(0.35))
model_1.add(Conv1D(128, 3, activation='relu'))
model_1.add(GlobalMaxPooling1D())
'''
model_2 = Sequential()
model_2.add(embedding_layer_1)
'''
model_2.add(SpatialDropout1D(0.35))
model_2.add(Conv1D(128, 3, activation='relu'))
model_2.add(GlobalMaxPooling1D())
'''

merged = Merge([model_1, model_2], mode='concat', concat_axis=-1)
model = Sequential()
model.add(merged)
model.add(Reshape((2, max_weibo_length, embedding_dim)))
model.add(Conv2D(128, 5, activation='relu', border_mode='valid', data_format='channels_first'))
model.add(MaxPooling2D(data_format='channels_first', pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(2))
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
ma_f_max = 0
for i in range(1):
    model.fit([X_train, X_train], y_train, validation_data=([X_valid, X_valid], y_vaild), epochs=3,
              batch_size=16, callbacks=[history])

    # del model
    # model = load_model('my_model_13.h5')
    score = model.evaluate([X_test, X_test], y_test)

    y_p = model.predict_classes([X_test, X_test])
    test_label = np.array(test_label)
    print
    print test_label[:30]
    print y_p[:30]
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

    sys_cor = dict()
    for i in range(2):
        sys_cor[i] = 0
    for x, y in zip(y_p, test_label):
        if x == y:
            sys_cor[x] += 1
    print sys_cor[1], sum(y_p), sum(test_label)
    preci = sys_cor[1] / sum(y_p)
    recal = sys_cor[1] / sum(test_label)
    f1 = 2 * preci * recal / (preci + recal)
    print 'f1: ', f1  # 0.7600

    '''
    file = './subjective_weibo.txt'
    f = open(file, 'w')
    for i, k in enumerate(list(y_p)):
        if k == 1:
            f.write('%d\n' % i)
    f.close()
    '''
    '''
    system_correct = dict()
    for i in range(7):
        system_correct[i] = 0

    for x, y in zip(y_p, test_label):
        if x == y:
            system_correct[x] += 1

    print system_correct.keys()
    print system_correct.values()
    system_proposed = collections.Counter(y_p)
    print system_proposed
    gold = collections.Counter(test_label)
    print gold
    sum_p = 0
    sum_r = 0
    sum_sys_cor = 0
    sum_sys_pro = 0
    sum_gold = 0
    for i in range(7):
        sum_p += system_correct[i] / (system_proposed[i] + K.epsilon())
        sum_r += system_correct[i] / gold[i]
        sum_sys_cor += system_correct[i]
        sum_sys_pro += system_proposed[i]
        sum_gold += gold[i]

    ma_p = sum_p / 7
    ma_r = sum_r / 7
    ma_f = 2 * ma_p * ma_r / (ma_p + ma_r)
    print 'ma_f: ', ma_f
    mi_p = sum_sys_cor / sum_sys_pro
    mi_r = sum_sys_cor / sum_gold
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r)
    print 'mi_f: ', mi_f
'''
'''
    if ma_f > ma_f_max:
        model.save('my_model_13.h5')
        ma_f_max = ma_f
    print ma_f_max
'''
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
