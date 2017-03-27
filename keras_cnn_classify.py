# coding=utf-8

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers import Dropout, Dense, Activation
from keras.layers import Embedding
from keras.callbacks import Callback, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

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


def load_train_data(filename):
    f = open(filename, 'r')
    train = []
    label = []
    for line in f.readlines()[:200000]:
        line = line.strip().decode('utf-8').split()
        train.append(line[:-1])
        t_label = int(line[-1])
        if t_label == 2:
            label.append(1)
        else:
            label.append(0)
    return train, label


def load_test_data(filename):
    f = open(filename, 'r')
    test = []
    label = []
    for line in f.readlines():
        line = line.strip().decode('utf-8').split()
        t_label = int(line[-1])
        if t_label == 1:
            continue
        elif t_label == 2:
            label.append(1)
        else:
            label.append(0)
        test.append(line[:-1])
    return test, label

train, train_label = load_train_data('./weibo/train_set.txt')
# test, test_label = load_test_data('./NLPCC/test_data_nlpcc14_weibo.txt')


def read_vec(filename):
    f = open(filename)
    vocabulary_size, embedding_dim = f.readline().split()
    embeddings = []
    words = []
    for line in f.readlines():
        words.append(line.split()[0])
        embeddings.append([float(num) for num in line.split()[1:]])
        # embeddings.append(line.split()[1:])
    f.close()
    return vocabulary_size, embedding_dim, embeddings, words

vocabulary_size, embedding_dim, embeddings, words = read_vec('vec_weibo_s.txt')
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
    global c
    for tweet in tweets:
        tweet_to_id = []
        for w in tweet:
            try:
                tweet_to_id.append(dictionary[w])
            except Exception as e:
                tweet_to_id.append(dictionary['UNK'])
        tweets_to_id.append(tweet_to_id)
    return tweets_to_id

dictionary = build_dict(words)
train_id = word_to_id(train, dictionary)
# test_id = word_to_id(test, dictionary)
print len(train_id)

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
train_num = 100000
vaild_num = train_num + 50000
max_weibo_length = 140

X_train = train_id[:train_num]
y_train = np.array(train_label[:train_num])
X_train = sequence.pad_sequences(X_train,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')
X_valid = train_id[train_num:vaild_num]
y_vaild = np.array(train_label[train_num:vaild_num])
X_valid = sequence.pad_sequences(X_valid,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')
X_test = train_id[vaild_num:]
y_test = np.array(train_label[vaild_num:])
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
model.add(Embedding(vocabulary_size,
                    embedding_dim,
                    weights=[embeddings],
                    trainable=False,
                    input_length=max_weibo_length,
                    dropout=0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

history = LossHistory()
model.fit(X_train, y_train, validation_data=(X_valid, y_vaild), nb_epoch=2,
          batch_size=32, callbacks=[history])

score = model.evaluate(X_test, y_test)

print model.metrics_names
print score
print "Test loss: ", score[0]
print "Test accuracy: ", score[1]
print len(history.losses)

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
