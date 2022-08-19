import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
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


tk = Tokenizer(num_words=None, char_level=True)

human_files = ['HumanBestNoRegCNNDropped.h5','HumanBestRegCNNDropped.h5','HumanBestNoRegParamsDropped.h5','HumanBestRegParamsDropped.h5', 'HumanBestNoRegCNNOverlapped.h5','HumanBestRegCNNOverlapped.h5','HumanBestNoRegParamsOverlapped.h5','HumanBestRegParamsOverlapped.h5']
human_models = []
for x in human_files:
    human_models.append(keras.models.load_model(x))

human_crms = pd.read_csv('HumanCRMValidationData.txt', delimiter='\t', header=None, names=['Name', 'Sequence', 'Label'])

human_seqs = np.array(human_crms.Sequence.tolist())
human_labels = np.array(human_crms.Label.tolist())
tk.fit_on_texts(human_seqs)
token_human = tk.texts_to_sequences(human_seqs)
human_samples = len(human_seqs)
human_onehot = OneHot(token_human)
human_onehot = np.array(human_onehot).reshape(human_samples, 1000, 4, 1).astype('float32')
for i in range(len(human_models)):
    human_predictions = human_models[i].predict(human_onehot, verbose=0, batch_size=100)
    col_name = human_files[i].split('.')
    human_crms[col_name[0]] = human_predictions
human_crms.to_csv('HumanFinalResults.tsv', sep="\t")