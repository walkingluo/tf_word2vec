from keras.preprocessing import sequence
from keras.layers import Convolution1D, Flatten, Dropout, Dense, LSTM, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
import numpy as np
from test_data import main
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
X, y, _dict, reverse_dict = main()
# print len(X), len(y), len(_dict), len(reverse_dict)


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

tweets_to_id = word_to_id(X, _dict)

# n = len(tweets_to_id)
train_num = 8000
vaild_num = 9000
print len(tweets_to_id)
# print tweets_to_id[:2]


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
vocabulary_size, embedding_dim, embeddings, words = read_vec('vec_s_t_l.txt')
print len(words), type(words[0])
print words[:5]
vocabulary_size = int(vocabulary_size)
embedding_dim = int(embedding_dim)
embeddings = np.array(embeddings)
count = 0
'''
for i in range(vocabulary_size):
    try:
        embeddings[i] = embeddings_index[words[i]]
    except Exception, e:
        embeddings[i] = np.zeros(embedding_dim)
        count += 1
        continue
'''
print count

max_weibo_length = 140
X_train = tweets_to_id[:train_num]
y_train = np.array(y[:train_num])
X_train = sequence.pad_sequences(X_train,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')
X_valid = tweets_to_id[train_num:vaild_num]
y_vaild = np.array(y[train_num:vaild_num])
X_valid = sequence.pad_sequences(X_valid,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')
X_test = tweets_to_id[vaild_num:]
y_test = np.array(y[vaild_num:])
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
model.add(Convolution1D(128, 5, border_mode='same', activation='relu'))
model.add(MaxPooling1D(5))
model.add(Convolution1D(128, 5, border_mode='same', activation='relu'))
model.add(MaxPooling1D(5))
model.add(Convolution1D(128, 5, border_mode='same', activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])
history = LossHistory()
model.fit(X_train, y_train, validation_data=(X_valid, y_vaild), nb_epoch=2,
          batch_size=128, callbacks=[history])

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
