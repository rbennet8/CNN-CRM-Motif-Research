from dataclasses import replace
from numpy import argmax
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from multiprocessing import Process
import time


'''
Converts tokenized sequences into one-hot matrices.
Data must be a list of lists, with each index being a value from 1-5.
'''
def OneHot(data):
    num_classes = 4
    new_data = []
    # For list in list/tokenized sequence in data
    for x in data:
        # Converting to numpy array and creating matrix to hold data
        class_vector = np.array(x)
        categorical = np.zeros(class_vector.shape+(num_classes,))
        for c in range(1,5,1):
            # Finds indexes where c (1-4) is and places the appropriate one-hot converstion at that index, in the categorical matrix
            categorical[np.where(class_vector == c)]=np.array([1 if i == c else 0.0 for i in range(1,5,1)])
        # Find occurance of N, replaces it with the appropriate conversion, the appends it to the list of data
        categorical[np.where(class_vector==5)]=[0.25]*4
        new_data.append(categorical)
    return new_data


'''
Loads model, creates simplified prediction model from it (for 1st convolution output), saves feature maps, extracts high performing kernels, and outputs MEME format files.
Combines motifs based on best performing kernels (highest 25% of values) per filter and row (1-3) in said filter.
    This is due to the nature of filters seeking a specific combination of 0s/1s in a kernel and the placement of 1s being crucial in one-hot data.
    Also, only motifs that are highly represented (100+ sequences) are output.
'''
def GetMotifs(model_path, one_hot, name, freq):
    # Loading the model and using it to create a prediction model that only outputs data form the first convolution
    main_model = keras.models.load_model(model_path)
    pred_model = keras.Model(inputs=main_model.inputs, outputs=main_model.layers[0].output)
    # Getting feature maps; arrays of how the kernels performed in each filter
    feature_maps = pred_model.predict(one_hot, verbose=0, batch_size=50)
    # Getting min/max and calculating cutoff (highest 25% of values) using those
    model_max = np.amax(feature_maps)
    model_min = np.min(feature_maps)
    cutoff = ((model_max - model_min) * 0.75) + model_min
    # Getting indexes and sequences of high performing kernels in each filter
    high_perf = {}
    zeros = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # Feature map shape is 4D; [sequence, iterations across x-axis, iterations across y-axis, filter]
    #   i.e. [n, 981, 3, 420]
    #       n is number of seqs
    #       981 iterations across x-axis; kernel's x dimension is 20 and sequence is 1000, so ~1000-20
    #       3 iterations across y-axis; kernel's y dimension is 2 and depth is 4 (because of one-hotting), so first 2, middle 2, bottom 2
    #       420 is number of filters that hold values for each kernel performance
    for a in range(feature_maps.shape[0]): # Iterating through each sequence
        for b in range(feature_maps.shape[-1]): # Iterating through the filters
            row_1 = []
            row_2 = []
            row_3 = []
            data = feature_maps[a,:,:,b] # Checking every filter, one sample at a time
            if np.amax(data) > cutoff: # Only accessing arrays that have values above the cutoff
                for c in range(data.shape[-1]): # Iterating through the 3 rows of the filter
                    temp_row = [x for x in range(len(data)) if data[x][c] > cutoff] # Only getting values above threshold
                    for d in temp_row:
                        # a is sequence index, b is filter number, c (0-2) is row index, and d (0-981) is column index
                        temp_seq = []
                        # One-hot shape is (n, 1000, 4, 1) - number of seqs, length, depth, num of values at index
                        # Have to reshape all the one hot sequences, because they're lists of lists
                        # i.e. [[1],[0],[0],[1]] reshaped to [1,0,0,1]
                        if c == 0:
                            temp_seq = [np.reshape(one_hot[a,d:d+20,c,:], 20)]
                            temp_seq.append(np.reshape(one_hot[a,d:d+20,c+1,:], 20))
                            temp_seq.append(zeros)
                            temp_seq.append(zeros)
                            row_1.append(temp_seq)
                        elif c == 1:
                            temp_seq = [zeros]
                            temp_seq.append(np.reshape(one_hot[a,d:d+20,c,:], 20))
                            temp_seq.append(np.reshape(one_hot[a,d:d+20,c+1,:], 20))
                            temp_seq.append(zeros)
                            row_2.append(temp_seq)
                        else:
                            temp_seq = [zeros, zeros]
                            temp_seq.append(np.reshape(one_hot[a,d:d+20,c,:], 20))
                            temp_seq.append(np.reshape(one_hot[a,d:d+20,c+1,:], 20))
                            row_3.append(temp_seq)
            # Keys are filter numbers and rows are broken into the indexes, so they aren't combined in motif output
            if not b in high_perf:
                high_perf[b] = [row_1, row_2, row_3]
            else:
                high_perf[b][0] += row_1
                high_perf[b][1] += row_2
                high_perf[b][2] += row_3
    # Creating the MEME format for output
    meme_output = {}
    for a in high_perf: # Iterating through filters
        row = -1
        for b in high_perf[a]: # Iterating through rows saved in each filter
            row += 1
            if len(b) > 100:
                # Creating empty matrix to build a nucleotide frequency matrix from
                added_seq = np.zeros([20, 4], dtype = float)
                for c in b:
                    added_seq = np.add(added_seq, np.array(c).T) # Had to transpose saved arrays to get the proper format
                meme_output[str(a)+'-'+str(row)] = np.divide(added_seq, len(b)) # Added Filter/Row key to dictionary, which equals corresponding motif/frequencies
    # Writing the data out in MEME format
    freq_vals = list(freq.values())
    with open(name+'-MEME.txt', 'w') as f:
        f.writelines(['MEME version 4\n', '\n', 'ALPHABET= ACGT\n', '\n', 'strands: + -\n', '\n', 'Background letter frequencies\n'])
        f.writelines(['A ', str(freq_vals[0]), ' C ', str(freq_vals[1]), ' G ', str(freq_vals[2]), ' T ', str(freq_vals[3]), '\n'])
        f.write('\n')
        for x in meme_output:
            f.writelines(['MOTIF ', name+'-Filter'+str(x), '\n'])
            f.write('letter-probability matrix: alength= 4 w= 20\n')
            for y in meme_output[x]:
                norm_bases = []
                if sum(y) >= 0.75: norm_bases = [x/sum(y) for x in y]
                else: norm_bases = [0.25, 0.25, 0.25, 0.25]
                f.writelines([str(norm_bases[0]), ' ', str(norm_bases[1]), ' ', str(norm_bases[2]), ' ', str(norm_bases[3]), '\n'])
            f.write('\n')


if __name__ == '__main__':
    start = time.time()
    print('- IMPORTING DATA -')
    # UNCOMMENT/COMMENT FOR SPECIFIC DATA SETS - - - - - - - - - - - - - - - -
    crm_path = 'HumanDroppedCRMSeqs.txt'
    # crm_path = 'HumanOverlappedCRMSeqs.txt'
    crm_seqs = pd.read_csv(crm_path, delimiter='\t', header=None, names=['Name', 'Sequence', 'Label'], dtype = {'Name': str, 'Sequence': str, 'Label': int}).sample(100000, random_state=0)
    crm_seqs = list(crm_seqs.Sequence)
    print('- DATA READ IN -')
    print('- TOKENIZING DATA AND GETTING NUCLEOTIDE FREQUENCIES -')
    # Using these to replace values and track frequency of nucleotides in the for loops below
    alph = ['A', 'C', 'G', 'T', 'N']
    token_vals = ['1', '2', '3', '4', '5']
    freq = {'A':0, 'C':0, 'G':0, 'T':0, 'N':0}
    # Iterating through the sequences and tokenizing them to specific values, so MEME output will be easier later
    for a in range(len(crm_seqs)):
        crm_seqs[a] = crm_seqs[a].upper()
        # Iterating through the determined alphabet, counting occurences of each, and replacing with associated integer as string
        for b in range(len(alph)):
            freq[alph[b]] += crm_seqs[a].count(alph[b])
            crm_seqs[a] = crm_seqs[a].replace(alph[b], token_vals[b])
        # Using list comprehension to change string into list of integers
        crm_seqs[a] = [int(b) for b in crm_seqs[a]]
    # Calculating frequencies of nucleotides
    freq_sum = sum(list(freq.values()))
    for a in freq:
        freq[a] = freq[a] / freq_sum
    print('- CREATING ONEHOT MATRIX -')
    # Getting one hot matrices and reshaping them for Keras
    one_hot = OneHot(crm_seqs)
    one_hot = np.array(one_hot).reshape(len(crm_seqs), 1000, 4, 1).astype('float32')
    print('- SEQUENCES PROCESSED -')
    # Setting paths for models
    # UNCOMMENT/COMMENT FOR SPECIFIC DATA SETS - - - - - - - - - - - - - - - -
    model_paths = ['HumanBestNoRegCNNDropped.h5', 'HumanBestNoRegParamsDropped.h5', 'HumanBestRegCNNDropped.h5', 'HumanBestRegParamsDropped.h5']
    # model_paths = ['HumanBestNoRegCNNOverlapped.h5', 'HumanBestNoRegParamsOverlapped.h5', 'HumanBestRegCNNOverlapped.h5', 'HumanBestRegParamsOverlapped.h5']
    print('- GETTING MOTIF SEQUENCES -')
    for x in model_paths:
        GetMotifs(x, one_hot, x.split('.')[0], freq)
    print('- MOTIFS OUTPUT -')
    end = time.time()
    print('Time elapsed -', (end-start))