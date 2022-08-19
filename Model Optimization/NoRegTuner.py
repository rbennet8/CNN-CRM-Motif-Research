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
from keras_tuner import Hyperband


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


def tune_cnn(hp):    
    filters_1 = hp.Int('filters_1', min_value=300, max_value=600, step=20, default=400)
    filters_2 = hp.Int('filters_2', min_value=300, max_value=600, step=20, default=400)
    filters_3 = hp.Int('filters_3', min_value=300, max_value=600, step=20, default=400)
    units = hp.Int('units', min_value=500, max_value=1200, step=50, default=800)
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    optimizer = keras.optimizers.Adam(learning_rate)
    model = Sequential()
    model.add(Conv2D(filters = filters_1, kernel_size = (20, 2), strides = 1, activation = 'relu', input_shape = (1000, 4, 1)))
    model.add(MaxPooling2D(pool_size = (10, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = filters_2, kernel_size = (15, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (7, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = filters_3, kernel_size = (5, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units = units, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model


print('Seed 1')

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
crms = pd.read_csv('DroppedCRMSeqs.txt', header = None, sep = '\t')
non_crms = pd.read_csv('DroppedNonCRMSeqs.txt', header = None, sep = '\t')
total_seqs = pd.concat([crms, non_crms], ignore_index = True)
total_seqs.columns = ['Names', 'Sequences', 'Labels']

x_tr, x_val, y_tr, y_val = train_test_split(total_seqs.Sequences.tolist(), total_seqs.Labels.tolist(), test_size=0.2)
x_tr, x_val, y_tr, y_val = np.array(x_tr), np.array(x_val), np.array(y_tr), np.array(y_val)
tk = Tokenizer(num_words=None, char_level=True)
tk.fit_on_texts(x_tr)
token_train = tk.texts_to_sequences(x_tr)
token_validate = tk.texts_to_sequences(x_val)
x_tr_samples = len(x_tr)
x_val_samples = len(x_val)
y_tr_samples = len(y_tr)
y_val_samples = len(y_val)
x_train = OneHot(token_train)
x_validate = OneHot(token_validate)
x_train = np.array(x_train).reshape(x_tr_samples, 1000, 4, 1).astype('float32')
x_validate = np.array(x_validate).reshape(x_val_samples, 1000, 4, 1).astype('float32')
y_train = y_tr.reshape(y_tr_samples, 1)
y_validate = y_val.reshape(y_val_samples, 1)

checkpoint_path = '/users/rbennet8/checkpoint-noreg'
checkpoint_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, monitor = 'val_accuracy', mode = 'max', save_best_only = True)
tuner = Hyperband(tune_cnn, max_epochs = 10, objective = 'val_accuracy', seed = 1, directory = 'Hyberband No-Reg Long', project_name = 'Tune No-Reg', overwrite = True)
tuner.search(x = x_train, y = y_train, epochs = 10, batch_size = 128, validation_data = (x_validate, y_validate), verbose = 2, callbacks = [checkpoint_callbacks])

print(tuner.get_best_models()[0].summary())
print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models = 1)[0]
print (model.summary())
loss, accuracy = model.evaluate(x_validate, y_validate)
print('loss:', loss)
print('accuracy:', accuracy)