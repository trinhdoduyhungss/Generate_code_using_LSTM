from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import requests
import re

text = (open("data_code.txt", encoding="utf8").read())
processed_text = text
print('corpus length:', len(processed_text))

chars = sorted(list(set(processed_text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 50
step = 1
sentences = []
next_chars = []
for i in range(0, len(processed_text) - maxlen, step):
    sentences.append(processed_text[i: i + maxlen])
    next_chars.append(processed_text[i + maxlen])
print('nb sequences:', len(sentences))
print('sentences:', sentences)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(200))
model.add(Dropout(0.1))
model.add(Dense(len(chars), activation='softmax'))

#optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(x, y,
          batch_size=100,
          epochs=100)
model.save_weights('code4_generator_newLSTM.h5')
