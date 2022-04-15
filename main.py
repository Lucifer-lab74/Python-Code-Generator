import numpy
import os
import sys
import time

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

import matplotlib.pyplot as plt

SEQ_LENGTH = 100
EPOCHS = 10
BATCH_SIZE = 128

text = (open("Data.txt", encoding="utf8").read().lower())

INPUT_FILE_LEN = len(text)

chars = sorted(list(set(text)))
VOCAB_LENGTH = len(chars)

print("Length of file: " + str(INPUT_FILE_LEN))
print("Total Vocab length (unique chars in input) : " + str(VOCAB_LENGTH))

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print(char_to_int)
print(int_to_char)

dataX = []
dataY = []
for i in range(0, INPUT_FILE_LEN - SEQ_LENGTH, 1):
    seq_in = text[i:i + SEQ_LENGTH]
    seq_out = text[i + SEQ_LENGTH]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

samples = len(dataX)
print("Total samples: " + str(samples))

X = numpy.reshape(dataX, (samples, SEQ_LENGTH, 1))

X = X / float(VOCAB_LENGTH)

y = np_utils.to_categorical(dataY)
print("X.shape=" + str(X.shape))
print("y.shape=" + str(y.shape))

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))  # 0.5
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

histroy = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
model.save("Model0.h5")
# get_ipython().run_line_magic('matplotlib', 'inline')
print(histroy.history.keys())

plt.plot(histroy.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

start = numpy.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print("input code starts with: [", ''.join([int_to_char[value] for value in pattern]), "]")
# generate characters
for i in range(500):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(VOCAB_LENGTH)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\n#####.")

model.save('python_code_generator.h5')
