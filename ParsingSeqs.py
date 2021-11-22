import pandas as pd
import numpy as np

# This function
def process(data):
    data[3] = np.nan
    for idx, _ in data.iterrows():
        data.loc[idx, 3] = data.loc[idx, 2] - data.loc[idx, 1]
    data = data.where(data[3] > 1000).dropna()
    data.columns = ['names', 'start', 'end', 'length']
    new_data = pd.DataFrame(columns = ['names', 'start', 'end', 'length'])
    for idx, _ in data.iterrows():
        end = data.loc[idx, 'end']
        start = data.loc[idx, 'start']
        length = data.loc[idx, 'length']
        num_seqs = int(length / 1000) # Number of divisions in the sample
        leftover = length % 1000 # Remainder of sequence
        mid_seq = int(num_seqs / 2) # Finding the middle of the divisions
        leftoverDiv = int(leftover / num_seqs) # Dividing leftover into equal parts
        if leftover > 0 and leftover < 800:
            for x in range(num_seqs):
                name = data.loc[idx, 'names']
                new_start = 0
                new_end = 0
                if x == mid_seq:
                    missed_bases = (1000 * num_seqs) - (leftoverDiv * num_seqs) # This calculates the 1-4 bases that were lost due to rounding
                    new_start = start + 1000*x + leftoverDiv*x + missed_bases
                    new_end = new_start + 1000
                elif leftover > 0:
                    new_start = start + 1000*x + leftoverDiv*x
                    new_end = new_start + 1000
                leng = new_end - new_start
                new_data.loc[len(new_data.index)] = [name, int(new_start), int(new_end), leng]
        elif leftover > 799:
            for x in range(num_seqs):
                name = data.loc[idx, 'names']
                new_start = 0
                new_end = 0
                if x == mid_seq:
                    new_start = start + 1000*x
                    new_end = new_start + leftover
                elif leftover != 0:
                    new_start = start + 1000*x
                    new_end = new_start + 1000
                leng = new_end - new_start
                new_data.loc[len(new_data.index)] = [name, int(new_start), int(new_end), leng]
        else:
            for x in range(num_seqs):
                name = data.loc[idx, 'names']
                new_start = start + 1000*x
                new_end = new_start + 1000
            new_data.loc[len(new_data.index)] = [name, int(new_start), int(new_end), 1000]
    new_data = new_data.drop('length', axis = 1)
    return new_data


crm_data = pd.read_csv('HumanCRMs.txt', header = None, sep = '\t')
crm_data = crm_data.sample(n = 100000, random_state = 5)
crm_data = process(crm_data)
crm_data.to_csv('TestCRMs.csv', sep='\t', index = False, header = False)
non_crm_data = pd.read_csv('HumanNonCRMs.txt', header = None, sep = '\t')
non_crm_data = non_crm_data.sample(n = 100000, random_state = 5)
non_crm_data = process(non_crm_data)
non_crm_data.to_csv('TestNonCRMs.csv', sep='\t', index = False, header = False)