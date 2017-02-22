from keras.preprocessing import sequence
from keras.layers import Convolution1D, Flatten, Dropout, Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import numpy as np
from test_data import main

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
train_num = 1400000
print len(tweets_to_id)
# print tweets_to_id[:2]


def read_vec(filename):
    f = open(filename)
    vocabulary_size, embedding_dim = f.readline().split()
    embeddings = []
    for line in f.readlines():
        embeddings.append([float(num) for num in line.split()[1:]])
        # embeddings.append(line.split()[1:])
    f.close()
    return vocabulary_size, embedding_dim, embeddings
vocabulary_size, embedding_dim, embeddings = read_vec('vec_st.txt')
vocabulary_size = int(vocabulary_size)
embedding_dim = int(embedding_dim)
embeddings = np.array(embeddings)

max_weibo_length = 140
X_train = tweets_to_id[:train_num]
y_train = np.array(y[:train_num])
X_train = sequence.pad_sequences(X_train,
                                 maxlen=max_weibo_length,
                                 padding='post',
                                 truncating='post')
X_test = tweets_to_id[train_num:]
y_test = np.array(y[train_num:])
X_test = sequence.pad_sequences(X_test,
                                maxlen=max_weibo_length,
                                padding='post',
                                truncating='post')
# print len(X_train)
# print len(X_test)

model = Sequential()
model.add(Embedding(vocabulary_size,
                    embedding_dim,
                    weights=[embeddings],
                    trainable=False,
                    input_length=max_weibo_length,
                    dropout=0.2))
model.add(Convolution1D(64, 3, border_mode='same'))
model.add(Convolution1D(32, 3, border_mode='same'))
model.add(Convolution1D(16, 3, border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=128)

score = model.evaluate(X_test, y_test)
print model.metrics_names
print score
print "Test loss: ", score[0]
print "Test accuracy: ", score[1]
