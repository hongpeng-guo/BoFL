import matplotlib as mpl
import matplotlib.pyplot as plt
import json, sys
sys.path.insert(1, '../')
import AGX.dvfs as agx
import TX2.dvfs as tx2
import numpy as np

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
markers = ['o','P','s','>','D','^']


Home_folder = '/Users/hongpengguo/Desktop/Spring22/Middleware22/figures'

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}


def autolabel(rects, ax, bit=2, pos='center'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height % 1 == 0:
            ax.annotate(height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", size=15,
                        ha=pos, va='bottom')
        elif bit == 2:
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", size=15,
                        ha=pos, va='bottom')
        else:
            ax.annotate('{:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", size=15,
                        ha=pos, va='bottom')


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


# Plot the first motivation figure to show non-linear, monotonic relation

c1, c2 = agx.CPU_FREQ_TABLE[0], agx.CPU_FREQ_TABLE[-1]
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
axs[0].set_title('(a) Execution Latency per Minibatch', fontsize='14', fontweight='bold')
axs[0].plot(x, t2, marker=markers[0], markersize=10, label='CPU Frequency: 2.2 GHz', color=colors[3])
axs[0].plot(x, t3, marker=markers[1], markersize=10, label='CPU Frequency: 0.4 GHz', color=colors[4])
axs[0].set_ylabel('Second', fontsize='16')
axs[0].set_yticks([0.3, 0.4])
axs[0].grid(linestyle='--', linewidth=2)

axs[1].set_title('(b) Energy Consumption per Minibatch', fontsize='14', fontweight='bold')
axs[1].plot(x, e2, marker=markers[0], markersize=10, label='CPU Frequency: 2.2 GHz', color=colors[3])
axs[1].plot(x, e3, marker=markers[1], markersize=10, label='CPU Frequency: 0.4 GHz', color=colors[4])
axs[1].legend(fancybox=True, fontsize='16')
axs[1].set_xlabel('GPU Frequencies (GHz)', fontsize='16')
axs[1].set_ylabel('Joule', fontsize='16')
axs[1].grid(linestyle='--', linewidth=2)
plt.savefig('{}/{}.pdf'.format(Home_folder, 'motivation1'), 
            dpi=600, pad_inches = 0, bbox_inche='tight')



# Plot the second motivation figure to show the differences between models

c1, c2 = tx2.CPU_FREQ_TABLE[4], tx2.CPU_FREQ_TABLE[-1]
m1, m2 = tx2.EMC_FREQ_TABLE[0], tx2.EMC_FREQ_TABLE[-1]
g1, g2 = tx2.GPU_FREQ_TABLE[0], tx2.GPU_FREQ_TABLE[-1]

t1, e1 = [], []
t2, e2 = [], []
t3, e3 = [], []
x = tx2.CPU_FREQ_TABLE[2:-2]
for c in x:
    k = (c, g2, m2)
    t1.append(Data[TX2][CIFAR10][k][0])
    e1.append(Data[TX2][CIFAR10][k][1])
    t2.append(Data[TX2][ImageNet][k][0])
    e2.append(Data[TX2][ImageNet][k][1])
    t3.append(Data[TX2][IMDB][k][0])
    e3.append(Data[TX2][IMDB][k][1])

fig, axs = plt.subplots(2, sharex=True)
x = [1.0 * i / 1e6 for i in x]
axs[0].set_title('(a) Execuation Latency per Minibatch', fontsize='14', fontweight='bold')
axs[0].plot(x, t1, marker=markers[2], markersize=10, label='ViT', color=colors[0])
axs[0].plot(x, t2, marker=markers[3], markersize=10, label='ResNet50', color=colors[1])
axs[0].plot(x, t3, marker=markers[4], markersize=10, label='LSTM', color=colors[2])
axs[0].set_ylabel('Second', fontsize='16')
axs[0].grid(linestyle='--', linewidth=2)


axs[1].set_title('(b) Energy Consumption per Minibatch', fontsize='14', fontweight='bold')
axs[1].plot(x, e1, marker=markers[2], markersize=10, label='ViT', color=colors[0])
axs[1].plot(x, e2, marker=markers[3], markersize=10, label='ResNet50', color=colors[1])
axs[1].plot(x, e3, marker=markers[4], markersize=10, label='LSTM', color=colors[2])
axs[1].set_xlabel('CPU Frequencies (GHz)', fontsize='16')
axs[1].set_ylabel('Joule', fontsize='16')
axs[1].legend(ncol=3, fancybox=True, fontsize='16', loc='best', bbox_to_anchor=(0., 0., 1.0, 0.5))
axs[1].set_yticks([5.0, 7.0, 9.0])
axs[1].set_xticks([0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
axs[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.1f}'))
axs[1].grid(linestyle='--', linewidth=2)
plt.savefig('{}/{}.pdf'.format(Home_folder, 'motivation2'), 
            dpi=600, pad_inches = 0, bbox_inche='tight')


# Plot the third motivation figure to show the differences between devices

ViT_ratio = [ list(Data[AGX][CIFAR10].values())[-1][0] / list(Data[TX2][CIFAR10].values())[-1][0],
                list(Data[AGX][CIFAR10].values())[-1][1] / list(Data[TX2][CIFAR10].values())[-1][1]]
ResNet_ratio = [ list(Data[AGX][ImageNet].values())[-1][0] / list(Data[TX2][ImageNet].values())[-1][0],
                list(Data[AGX][ImageNet].values())[-1][1] / list(Data[TX2][ImageNet].values())[-1][1]]
LSTM_ratio = [ list(Data[AGX][IMDB].values())[-1][0] / list(Data[TX2][IMDB].values())[-1][0],
                list(Data[AGX][IMDB].values())[-1][1] / list(Data[TX2][IMDB].values())[-1][1]]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
labels = ['ViT', 'ResNet50', 'LSTM']
x = np.arange(len(labels))

ax1.set_title('(a) Execution Latency', fontsize='14', fontweight='bold')
ax1.set_xticks(x)
ax1.set_ylim(0.0, 1.05)
ax1.set_yticks([0.2, 0.6, 1.0])
ax1.set_yticklabels(['0.2 x', '0.6 x', '1.0 x'])
ax1.set_xticklabels(labels)
rect1 = ax1.bar(0, ViT_ratio[0], width=0.5, hatch=None, edgecolor='black', color=colors[0])
rect2 = ax1.bar(1, ResNet_ratio[0], width=0.5, hatch='/', edgecolor='black', color=colors[1])
rect3 = ax1.bar(2, LSTM_ratio[1], width=0.5, hatch='\\', edgecolor='black', color=colors[2])
ax1.yaxis.grid(linestyle='--',  linewidth=1.5)
ax1.set_ylabel('Normalized Performace to TX2', fontsize='16')
ax1.text(0.5, 0.78, "TX2", ha="left", va="bottom", rotation=90, size=18,
            bbox=dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="black", lw=2))
tx2_line = ax1.get_ygridlines()[2]
tx2_line.set_color('red')
tx2_line.set_linewidth(4)
autolabel(rect1, ax1)
autolabel(rect2, ax1)
autolabel(rect3, ax1)


ax2.set_title('(b) Energy Consumption', fontsize='14', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
rect4 = ax2.bar(0, ViT_ratio[1], width=0.5, hatch=None, edgecolor='black', color=colors[0])
rect5 = ax2.bar(1, ResNet_ratio[1], width=0.5, hatch='/', edgecolor='black', color=colors[1])
rect6 = ax2.bar(2, LSTM_ratio[1], width=0.5, hatch='\\', edgecolor='black', color=colors[2])
ax2.yaxis.grid(linestyle='--',  linewidth=1.5)
tx2_line = ax2.get_ygridlines()[2]
tx2_line.set_color('red')
tx2_line.set_linewidth(4)
autolabel(rect4, ax2)
autolabel(rect5, ax2)
autolabel(rect6, ax2)


plt.savefig('{}/{}.pdf'.format(Home_folder, 'motivation3'), 
            dpi=600, pad_inches = 0, bbox_inche='tight')
