# coding=utf-8

from __future__ import division
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation, SpatialDropout1D, Merge, Reshape
from keras.layers import Embedding
from keras.callbacks import Callback
from keras.models import load_model
import keras.backend as K
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
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


train, train_label = subjective_classify('./NLPCC/1/train_data_nlpcc13.txt')
test, test_label = subjective_classify('./NLPCC/1/test_data_nlpcc13.txt')
# train, train_label = load_train_data('./NLPCC/1/train_data_nlpcc13.txt')
# test, test_label = load_test_data('./NLPCC/1/test_data_nlpcc13.txt')
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

vocabulary_size, embedding_dim, embeddings, words = read_vec('wordvec_final.txt')
vocabulary_size = int(vocabulary_size)
embedding_dim = int(embedding_dim)
embeddings = np.array(embeddings)
embeddings = np.round(embeddings, 6)
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

embedding_layer_static = Embedding(vocabulary_size,
                                   embedding_dim,
                                   weights=[embeddings],
                                   trainable=False,
                                   input_length=max_weibo_length)
embedding_layer_non_static = Embedding(vocabulary_size,
                                       embedding_dim,
                                       weights=[embeddings],
                                       trainable=True,
                                       input_length=max_weibo_length)
embedding_layer_rand = Embedding(vocabulary_size,
                                 embedding_dim,
                                 input_length=max_weibo_length)
drop_rate = 0.3

model_1 = Sequential()
model_1.add(embedding_layer_non_static)
model_1.add(SpatialDropout1D(drop_rate))
model_1.add(Conv1D(100, 3, activation='relu', padding='same'))
model_1.add(GlobalMaxPooling1D())

model_1_s = Sequential()
model_1_s.add(embedding_layer_static)
model_1_s.add(SpatialDropout1D(drop_rate))
model_1_s.add(Conv1D(100, 3, activation='relu', padding='same'))
model_1_s.add(GlobalMaxPooling1D())

model_2 = Sequential()
model_2.add(embedding_layer_non_static)
model_2.add(SpatialDropout1D(drop_rate))
model_2.add(Conv1D(100, 4, activation='relu', padding='same'))
model_2.add(GlobalMaxPooling1D())

model_2_s = Sequential()
model_2_s.add(embedding_layer_static)
model_2_s.add(SpatialDropout1D(drop_rate))
model_2_s.add(Conv1D(100, 4, activation='relu', padding='same'))
model_2_s.add(GlobalMaxPooling1D())

model_3 = Sequential()
model_3.add(embedding_layer_non_static)
model_3.add(SpatialDropout1D(drop_rate))
model_3.add(Conv1D(100, 5, activation='relu', padding='same'))
model_3.add(GlobalMaxPooling1D())

model_3_s = Sequential()
model_3_s.add(embedding_layer_static)
model_3_s.add(SpatialDropout1D(drop_rate))
model_3_s.add(Conv1D(100, 5, activation='relu', padding='same'))
model_3_s.add(GlobalMaxPooling1D())

merged = Merge([model_1, model_2, model_3, model_1_s, model_2_s, model_3_s], mode='concat', concat_axis=-1)
model = Sequential()
model.add(merged)
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

history = LossHistory()
ma_f_max = 0
model.fit([X_train, X_train], y_train, validation_data=([X_valid, X_valid], y_vaild), epochs=2,
          batch_size=54, callbacks=[history])

# del model
# model = load_model('my_model_13.h5')
# score = model.evaluate([X_test, X_test], y_test)

y_p = model.predict_classes([X_test, X_test])
test_label = np.array(test_label)
'''
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
        system_correct[y] += 1

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
