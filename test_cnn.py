import pandas as pd
import random
import numpy as np

from numpy import argmax
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization


def GenSeqs():
    alphabet = 'GATC'
    bindingSeq = 'GGAATTCCTTAAGGAATTCCTTAAGGAATTCCTTAAGG'
    withBinding = 1000 - len(bindingSeq)

    totalSeqs = pd.DataFrame(columns=['Names', 'Sequences', 'Labels'])

    for x in range(2500):
        crm = []
        seq = ''
        for i in range(withBinding + 1):
            if (i == withBinding / 2):
                seq = seq + bindingSeq
            else:
                seq = seq + alphabet[random.randint(0, 3)]
        crm.append('CRM seq ' + str(x + 1))
        crm.append(seq)
        crm.append(1)
        totalSeqs.loc[len(totalSeqs)] = crm

    for x in range(2500):
        nonCrm = []
        seq = ''
        for i in range(1000):
            seq = seq + alphabet[random.randint(0, 3)]
        nonCrm.append('Non-CRM seq ' + str(x + 1))
        nonCrm.append(seq)
        nonCrm.append(0)
        totalSeqs.loc[len(totalSeqs)] = nonCrm

    print(totalSeqs)

    return totalSeqs


def OneHot(data):
    num_classes = 4
    new_data = []

    for x in data:
        class_vector = np.array(x)
        categorical = np.zeros(class_vector.shape + (num_classes,))
        for c in range(1, 5, 1):
            categorical[np.where(class_vector == c)] = np.array([1 if i == c else 0.0 for i in range(1, 5, 1)])
        new_data.append(categorical)

    return new_data


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Returns Pandas dataframe of names, sequences, and labels
totalSeqs = GenSeqs()

# Splitting data and labels in to train/validation sets
x_tr, x_val, y_tr, y_val = train_test_split(totalSeqs.Sequences.tolist(), totalSeqs.Labels.tolist(), test_size = 0.1)
x_tr, x_val, y_tr, y_val = np.array(x_tr), np.array(x_val), np.array(y_tr), np.array(y_val)

# Tokenizing sequences
tk = Tokenizer(num_words=None, char_level=True)
tk.fit_on_texts(x_tr)
tokenTrain = tk.texts_to_sequences(x_tr)
tokenValidate = tk.texts_to_sequences(x_val)

# Onehot encoding tokenized sequences
oneHotTrain = OneHot(tokenTrain)
oneHotValidate = OneHot(tokenValidate)

# Resizing to fit Conv2D and making sure there aren't any array/list conflicts
oneHotTrain = np.array(oneHotTrain).reshape(4500, 1000, 4, 1).astype('float32')
oneHotValidate = np.array(oneHotValidate).reshape(500, 1000, 4, 1).astype('float32')

trainLabels = y_tr.reshape(4500, 1)
validateLabels = y_val.reshape(500, 1)

model = Sequential()
model.add(Conv2D(32, (4, 1), activation='relu', input_shape=(1000, 4, 1)))
model.add(MaxPooling2D((1, 1)))
model.add(Conv2D(64, 4, activation='relu'))
model.add(MaxPooling2D(1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("TRAINING:")
final = model.fit(oneHotTrain, trainLabels, batch_size=100, epochs=5, verbose=1)

print("VALIDATION:")
validate = model.evaluate(oneHotValidate, validateLabels, verbose=1)