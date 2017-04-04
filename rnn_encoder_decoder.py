# https://gist.github.com/rouseguy/1122811f2375064d009dac797d59bae9

import numpy as np
from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.utils import np_utils
from random import randint


x_seq_length = 7
y_seq_length = 4
time_steps = 7
data_size = 10000
test_data_size = 30
epochs = 65

hidden_size = 128
batch_size = 128
layers = 1


# ==============================================================================
# mapping rule
#
#   X        =>    Y
#   ----------------
#   11+21    =>   32
#   132+254  =>  386
#   245+12   =>  257
#   ......
#
# ==============================================================================
def create_data(data_size, X_data_length, Y_data_length):
    X_data, Y_data = [], []
    for i in range(data_size):
        a = randint(0, 999)
        b = randint(0, 999)
        x = '%i+%i' % (a, b)
        x = x + (' ' * (X_data_length - len(x)))

        y = '%i' % (a + b)
        y = y + (' ' * (Y_data_length - len(y)))
        X_data.append(x)
        Y_data.append(y)
    return X_data, Y_data


def create_vocabulary(sentences):
    vocab = {}
    for sentence in sentences:
        for i in range(len(sentence)):
            ch = sentence[i]
            if ch in vocab:
                vocab[ch] += 1
            else:
                vocab[ch] = 1
    vocab_rev = sorted(vocab, key=vocab.get, reverse=True)
    vocab = dict([(x, y) for (y, x) in enumerate(vocab_rev)])
    return vocab, vocab_rev


def sentences_to_token_ids(sentences, vocabulary):
    data = []
    for sentence in sentences:
        characters = [sentence[i:i+1] for i in range(0, len(sentence), 1)]
        data.append([vocabulary.get(w) for w in characters])
    return data


def token_ids_to_sentence(token_ids, vocabulary_rev):
    lst = [vocabulary_rev[w] for w in token_ids]
    str = ''.join(lst)
    return str


# ==============================================================================
X_train, Y_train = create_data(data_size, x_seq_length, y_seq_length)

X_vacab, X_vacab_rev = create_vocabulary(X_train)
Y_vacab, Y_vacab_rev = create_vocabulary(Y_train)
X_CLASSES = len(X_vacab)
Y_CLASSES = len(Y_vacab)

X_train_ids = sentences_to_token_ids(X_train, X_vacab)
Y_train_ids = sentences_to_token_ids(Y_train, Y_vacab)

X_train_ids = np_utils.to_categorical(X_train_ids, num_classes=X_CLASSES)
X_train_ids = np.reshape(X_train_ids, (-1, x_seq_length, X_CLASSES))
Y_train_ids = np_utils.to_categorical(Y_train_ids, num_classes=Y_CLASSES)
Y_train_ids = np.reshape(Y_train_ids, (-1, y_seq_length, Y_CLASSES))


# ==============================================================================
print('Build model...')
model = Sequential()
model.add(LSTM(hidden_size, input_shape=(x_seq_length, X_CLASSES), return_sequences=False))
model.add(RepeatVector(y_seq_length))
for _ in range(layers):
    model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(Y_CLASSES)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_ids[:-test_data_size], Y_train_ids[:-test_data_size], epochs=epochs)


# ==============================================================================
X_test = X_train_ids[-test_data_size:]
Y_test = Y_train_ids[-test_data_size:]

predictions = model.predict(X_test, verbose=0)

for i, prediction in enumerate(predictions):
    x_ids = np.argmax(X_test[i], axis=1)
    x_str = token_ids_to_sentence(x_ids, X_vacab_rev)

    y_ids = np.argmax(Y_test[i], axis=1)
    y_str = token_ids_to_sentence(y_ids, Y_vacab_rev)

    pred_ids = np.argmax(prediction, axis=1)
    pred_str = token_ids_to_sentence(pred_ids, Y_vacab_rev)

    if y_str == pred_str:
        success = 'Yes'
    else:
        success = 'No'

    print('X: %s, Y: %s, Prediction: %s, Is OK? %s' % (x_str, y_str, pred_str, success))
