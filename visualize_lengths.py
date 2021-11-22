import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def lengths(data):
    lengths = []
    for idx, _ in data.iterrows():
        lengths.append(data.loc[idx, 2] - data.loc[idx, 1])
    return lengths


def plot_dict(lengths, data_lengths):
    plot_dict = {}
    for i in lengths:
        filter1 = data_lengths["Length"] > i
        filter2 = data_lengths["Length"] < i + 500
        val = data_lengths.where(filter1 & filter2).dropna()
        plot_dict[i] = len(val['Length'].tolist())
    return plot_dict


path_1 = "HumanNonCRMs.txt"
data_1 = pd.read_csv(path_1, header=None, sep='\t')
data_1[3] = lengths(data_1)
data_1.columns = ['Names', 'Start', 'End', 'Length']
shortest = data_1["Length"]>1000
longest = data_1["Length"]<10000
data_1 = data_1.where(longest).dropna()
data_1 = data_1.sort_values(by='Length')
parsed_1 = data_1.where(shortest).dropna()
parsed_1 = parsed_1['Length'].tolist()
parsed_1.sort()

path_2 = "HumanCRMs.txt"
data_2 = pd.read_csv(path_2, header=None, sep='\t')
data_2[3] = lengths(data_2)
data_2.columns = ['Names', 'Start', 'End', 'Length']
shortest = data_2["Length"]>1000
longest = data_2["Length"]<10000
data_2 = data_2.where(longest).dropna()
data_2 = data_2.sort_values(by='Length')
parsed_2 = data_2.where(shortest).dropna()
parsed_2 = parsed_2['Length'].tolist()
parsed_2.sort()

fig, ax = plt.subplots(figsize=(20, 15))
ax.hist(parsed_1, bins='auto', histtype='step', stacked=True, fill=False, linewidth=3, label='Non-CRM')
ax.hist(parsed_2, bins='auto', histtype='step', stacked=True, fill=False, linewidth=3, label='CRM')
ax.legend(loc='upper right', fontsize='xx-large')
plt.savefig('histogram.png')

iter_val = 1000
iters = int(max(parsed_1[-1], parsed_2[-1]) / 500)
lengths = []
for i in range(iters):
    lengths.append(iter_val)
    iter_val = iter_val + 500
data_lengths = pd.DataFrame(parsed_1, columns=['Length'])
plot_dict_1 = plot_dict(lengths, data_lengths)
data_lengths = pd.DataFrame(parsed_2, columns=['Length'])
plot_dict_2 = plot_dict(lengths, data_lengths)
x_1 = list(plot_dict_1.keys())
y_1 = list(plot_dict_1.values())
y_2 = list(plot_dict_2.values())
x_axis = np.arange(len(x_1))
plt.figure(figsize=(20, 15))
plt.bar(x_axis - 0.2, y_1, 0.4, label='Non-CRMs')
plt.bar(x_axis + 0.2, y_2, 0.4, label='CRMs')
plt.xticks(x_axis, x_1, rotation='vertical')
plt.legend(fontsize='xx-large', )
plt.show()
plt.savefig('bar.png')
plt.clf()

fig, ax = plt.subplots(figsize=(20, 15))
ax.hist(data_1['Length'].tolist(), bins='auto', histtype='step', stacked=True, fill=False, linewidth=3, label='Non-CRM')
ax.hist(data_2['Length'].tolist(), bins='auto', histtype='step', stacked=True, fill=False, linewidth=3, label='CRM')
ax.legend(loc='upper right', fontsize='xx-large')
plt.savefig('histogram2.png')

iters = iters + 2
iter_val = 0
lengths = []
for i in range(iters):
    lengths.append(iter_val)
    iter_val = iter_val + 500
data_lengths = pd.DataFrame(data_1["Length"])
plot_dict_3 = plot_dict(lengths, data_lengths)
data_lengths = pd.DataFrame(data_2["Length"])
plot_dict_4 = plot_dict(lengths, data_lengths)
x_1 = list(plot_dict_3.keys())
y_1 = list(plot_dict_3.values())
y_2 = list(plot_dict_4.values())
x_axis = np.arange(len(x_1))
plt.figure(figsize=(20, 15))
plt.bar(x_axis - 0.2, y_1, 0.4, label='Non-CRMs')
plt.bar(x_axis + 0.2, y_2, 0.4, label='CRMs')
plt.xticks(x_axis, x_1, rotation='vertical')
plt.legend(fontsize='xx-large', )
plt.show()
plt.savefig('bar2.png')
plt.clf()

print('=== Non-CRM lengths above 1000 ===')
print(plot_dict_1)
print()
print('=== CRM lengths above 1000 ===')
print(plot_dict_2)
print()
print('=== All non-CRM lengths ===')
print(plot_dict_3)
print()
print('=== All CRM lengths ===')
print(plot_dict_4)
