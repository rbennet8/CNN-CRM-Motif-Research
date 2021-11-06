import pandas as pd


def parse(data):
    filtered = pd.DataFrame(columns=['Names', 'Start', 'End')
    for idx, _ in data.iterrows():
        len = data.loc[idx, 2] - data.loc[idx, 1]
        if len > 1000:
            filtered = filtered.append({'Names': data.loc[idx, 0], 'Start': data.loc[idx, 1], 'End': data.loc[idx, 2]}, ignore_index=True)
    return filtered


print("=== HANDLING NON-CRM DATA ===")
path_1 = "HumanNonCRMs.txt"
data_1 = pd.read_csv(path_1, header = None, sep = '\t')
print(data_1)
data_1_parsed = parse(data_1)
print("=== NON-CRM DATA ===")
print(data_1_parsed)


print("=== HANDLING CRM DATA ===")
path_2 = "HumanCRMs.txt"
data_2 = pd.read_csv(path_2, header = None, sep = '\t')
print(data_2)
data_2_parsed = parse(data_2)
print("=== CRM DATA ===")
print(data_2_parsed)

print("=== SAVING DATA ===")
data_1_parsed.to_csv("HumanNonCRMs1000.csv", sep='\t', index = False, header = False)
data_2_parsed.to_csv("HumanCRMs1000.csv", sep='\t', index = False, header = False)