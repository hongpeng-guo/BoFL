import matplotlib.pyplot as plt
import json, sys
sys.path.insert(1, '../')
import AGX.dvfs as agx
import TX2.dvfs as tx2

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}


CIFAR10, ImageNet, IMDB = 0, 1, 2
TX2, AGX = 0, 1

TX2_CIFAR10_data = load_json('../TX2/CIFAR10.json')
TX2_ImageNet_data = load_json('../TX2/ImageNet.json')
TX2_IMDB_data = load_json('../TX2/IMDB.json')
AGX_CIFAR10_data = load_json('../AGX/CIFAR10.json')
AGX_ImageNet_data = load_json('../AGX/ImageNet.json')
AGX_IMDB_data = load_json('../AGX/IMDB.json')

Data = [[TX2_CIFAR10_data, TX2_ImageNet_data, TX2_IMDB_data],
        [AGX_CIFAR10_data, AGX_ImageNet_data, AGX_IMDB_data]]

'''
Plot the first motivation figure to show non-linear, monotonic relation
'''

c1, c2 = agx.CPU_FREQ_TABLE[4], agx.CPU_FREQ_TABLE[-1]
m1, m2 = agx.EMC_FREQ_TABLE[0], agx.EMC_FREQ_TABLE[-1]
g1, g2 = agx.GPU_FREQ_TABLE[0], agx.GPU_FREQ_TABLE[-1]

t1, e1 = [], []
t2, e2 = [], []
t3, e3 = [], []
x = agx.GPU_FREQ_TABLE[7:-1]
for g in x:
    k1 = (c1, g, m1)
    k2 = (c2, g, m2)
    k3 = (c1, g, m2)
    t1.append(Data[AGX][ImageNet][k1][0])
    e1.append(Data[AGX][ImageNet][k1][1])
    t2.append(Data[AGX][ImageNet][k2][0])
    e2.append(Data[AGX][ImageNet][k2][1])
    t3.append(Data[AGX][ImageNet][k3][0])
    e3.append(Data[AGX][ImageNet][k3][1])

fig, axs = plt.subplots(2, sharex=True)
x = [1.0 * i / 1e9 for i in x]
axs[0].plot(x, t2, marker='*', label='CPU Frequency: 0.4 GHz')
axs[0].plot(x, t3, marker='o', label='CPU Frequency: 2.2 GHz')
axs[1].plot(x, e2, marker='*', label='CPU Frequency: 0.4 GHz')
axs[1].plot(x, e3, marker='o', label='CPU Frequency: 2.2 GHz')
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend( lines, labels, loc = 'upper center', ncol=2 )
plt.show()


'''
Plot the second motivation figure to show the differences between models
'''
c1, c2 = tx2.CPU_FREQ_TABLE[4], tx2.CPU_FREQ_TABLE[-1]
m1, m2 = tx2.EMC_FREQ_TABLE[0], tx2.EMC_FREQ_TABLE[-1]
g1, g2 = tx2.GPU_FREQ_TABLE[0], tx2.GPU_FREQ_TABLE[-1]

t1, e1 = [], []
t2, e2 = [], []
t3, e3 = [], []
x = tx2.CPU_FREQ_TABLE[2:-2]
for c in x:
    k = (c, g2, m2)
    t1.append(Data[TX2][ImageNet][k][0])
    e1.append(Data[TX2][ImageNet][k][1])
    t2.append(Data[TX2][CIFAR10][k][0])
    e2.append(Data[TX2][CIFAR10][k][1])
    t3.append(Data[TX2][IMDB][k][0])
    e3.append(Data[TX2][IMDB][k][1])

fig, axs = plt.subplots(2, sharex=True)
x = [1.0 * i / 1e6 for i in x]
axs[0].plot(x, t1, marker='*', label='ResNet50')
axs[0].plot(x, t2, marker='o', label='ViT')
axs[0].plot(x, t3, marker='s', label='LSTM')
axs[1].plot(x, e1, marker='*', label='ResNet50')
axs[1].plot(x, e2, marker='o', label='ViT')
axs[1].plot(x, e3, marker='s', label='LSTM')
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend( lines, labels, loc = 'upper center', ncol=3)
plt.show()

