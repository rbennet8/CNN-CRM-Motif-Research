from numpy import argmax
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization


def OneHot(data):
    num_classes = 4
    new_data = []
    for x in data:
        class_vector = np.array(x)
        categorical = np.zeros(class_vector.shape + (num_classes,))
        for c in range(1, 5, 1):
            categorical[np.where(class_vector == c)] = np.array([1 if i == c else 0.0 for i in range(1, 5, 1)])
        categorical[np.where(class_vector == 5)] = [0.25] * 4
        new_data.append(categorical)
    return new_data

def OpenFiles(file, label):
    data = open(file, 'r')
    totalSeqs = pd.DataFrame(columns = ['Names', 'Sequences', 'Labels'])
    for line in data:
        if line.startswith('>'):
            tempList = []
            tempList.append(line.rstrip('\r\n'))
        else:
            tempList.append(line.upper().rstrip('\r\n'))
            tempList.append(label)
            totalSeqs.loc[len(totalSeqs)] = tempList
    return totalSeqs

def RunCNN(seqs):
    x_tr, x_val, y_tr, y_val = train_test_split(seqs.Sequences.tolist(), seqs.Labels.tolist(), test_size=0.2)
    x_tr, x_val, y_tr, y_val = np.array(x_tr), np.array(x_val), np.array(y_tr), np.array(y_val)
    tk = Tokenizer(num_words=None, char_level=True)
    tk.fit_on_texts(x_tr)
    token_train = tk.texts_to_sequences(x_tr)
    token_validate = tk.texts_to_sequences(x_val)
    x_train = OneHot(token_train)
    x_train = np.array(x_train).reshape(4500, 1000, 4, 1).astype('float32')
    x_validate = OneHot(token_validate)
    x_validate = np.array(x_validate).reshape(500, 1000, 4, 1).astype('float32')
    y_train = y_tr.reshape(4500, 1)
    y_validate = y_val.reshape(500, 1)
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 1), activation='relu', input_shape=(1000, 4, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (4, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    print("=== TRAINING: ===")
    final = model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1)
    print("=== VALIDATION: ===")
    validate = model.evaluate(x_validate, y_validate, verbose=1)


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
crms = "HumanNonCRMs1000Seqs.txt"
non_crms = "HumanCRMs1000Seqs.txt"
total_seqs = []
crm = 0
non_crm = 1
crm_seqs = OpenFiles(crms, crm)
crm_seqs = crm_seqs.sample(n = 50000, random_state = 5) # For testing --------------------------------------
non_crm_seqs = OpenFiles(non_crms, non_crm)
non_crm_seqs = non_crm_seqs.sample(n = 50000, random_state = 5) # For testing ------------------------------
total_seqs.append(pd.concat([non_crm_seqs, crm_seqs], ignore_index = True))
total_seqs
RunCNN(total_seqs)