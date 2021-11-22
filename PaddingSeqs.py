import pandas as pd


def OpenFiles(file, label):
    data = open(file, 'r')
    total_seqs = pd.DataFrame(columns = ['Names', 'Sequences', 'Labels'])
    for line in data:
        if line.startswith('>'):
            temp_list = []
            temp_list.append(line.rstrip('\r\n'))
        else:
            temp_list.append(line.upper().rstrip('\r\n'))
            temp_list.append(label)
            total_seqs.loc[len(total_seqs)] = temp_list
    return total_seqs


def RefPadding(data):
    for idx, _ in data.iterrows():
        seq = data.iloc[idx, 1]
        length = len(seq)
        if length < 1000:
            leftover = 1000 - length
            half_leftover = leftover // 2
            beginning = seq[0:half_leftover]
            beginning = beginning[::-1]
            ending = seq[-(leftover - half_leftover):]
            ending = ending[::-1]
            data.iloc[idx, 1] = beginning + seq + ending
    return data


crms = OpenFiles('TestCRMSeqs.txt', 0)
non_crms = OpenFiles('TestNonCRMSeqs.txt', 1)
crms = RefPadding(crms)
non_crms = RefPadding(non_crms)
total_seqs = pd.concat([crms, non_crms], ignore_index = True)
total_seqs.to_csv('TestTotalSeqs.txt', sep = '\t', index = False, header = False)