from numpy import argmax
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras import regularizers


def OneHot(data):
    num_classes = 4
    new_data = []
    for x in data:
        class_vector = np.array(x)
        categorical = np.zeros(class_vector.shape+(num_classes,))
        for c in range(1,5,1):
            categorical[np.where(class_vector == c)]=np.array([1 if i == c else 0.0 for i in range(1,5,1)])
        categorical[np.where(class_vector==5)]=[0.25]*4
        new_data.append(categorical)
    return new_data


def RunCNN(seqs):
    x_tr, x_val, y_tr, y_val = train_test_split(seqs.Sequences.tolist(), seqs.Labels.tolist(), test_size = 0.2)
    x_tr, x_val, y_tr, y_val = np.array(x_tr), np.array(x_val), np.array(y_tr), np.array(y_val)
    tk = Tokenizer(num_words=None, char_level=True)
    tk.fit_on_texts(x_tr)
    token_train = tk.texts_to_sequences(x_tr)
    token_validate = tk.texts_to_sequences(x_val)
    # Getting data lengths to abstract reshaping
    x_tr_samples = len(x_tr)
    x_val_samples = len(x_val)
    y_tr_samples = len(y_tr)
    y_val_samples = len(y_val)
    # Handling sequence data
    x_train = OneHot(token_train) # Returns a 4000 samole list, each 1000x4
    x_train = np.array(x_train).reshape(x_tr_samples, 1000, 4, 1).astype('float32')
    x_validate = OneHot(token_validate)
    x_validate = np.array(x_validate).reshape(x_val_samples, 1000, 4, 1).astype('float32')
    y_train = y_tr.reshape(y_tr_samples, 1)
    y_validate = y_val.reshape(y_val_samples, 1)
    model = Sequential()
    model.add(Conv2D(filters = 500, kernel_size = (20, 2), strides = 1, activation = 'relu', input_shape = (1000, 4, 1)))
    model.add(MaxPooling2D(pool_size = (10, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 580, kernel_size = (15, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (7, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 340, kernel_size = (5, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1150, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.01)))
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0001)))
    # Changing learning rate; default is 0.001
    optimizer = keras.optimizers.Adam(lr = 0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    print("TRAINING:")
    final = model.fit(x_train, y_train, batch_size = 100, epochs = 6, verbose = 2)
    print("VALIDATION:")
    validate = model.evaluate(x_validate, y_validate, verbose=2)
    model.save('MouseBestRegCNNDropped.h5')


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
crm_path = r"MouseDroppedCRMSeqs.txt"
crms = pd.read_csv(crm_path, header=0, sep='\t', names=['Names', 'Sequences', 'Labels'])
noncrm_path = r"MouseDroppedNonCRMSeqs.txt"
noncrms = pd.read_csv(noncrm_path, header=0, sep='\t', names=['Names', 'Sequences', 'Labels'])
total_seqs = pd.concat([crms, noncrms], ignore_index = True)
RunCNN(total_seqs)