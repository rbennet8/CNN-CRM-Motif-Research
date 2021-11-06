path_1 = "HumanNonCRMs1000.csv"
data_1 = pd.read_csv(path_1, header = None, sep = '\t')
for idx, rows in data_1.iterrows():
    mid = int((data_1.loc[idx, 1] + data_1.loc[idx, 2]) / 2)
    data_1.loc[idx, 1] = mid - 500
    data_1.loc[idx, 2] = mid + 500
data_1.to_csv("HumanNonCRMs1000.csv", sep='\t', index = False, header = False)

path_2 = "HumanCRMs1000.csv"
data_2 = pd.read_csv(path_2, header = None, sep = '\t')
for idx, rows in data_2.iterrows():
    mid = int((data_2.loc[idx, 1] + data_2.loc[idx, 2]) / 2)
    data_2.loc[idx, 1] = mid - 500
    data_2.loc[idx, 2] = mid + 500
write_path_2 = "HumanCRMs1000.csv"
data_2.to_csv(write_path_2, sep='\t', index = False, header = False)