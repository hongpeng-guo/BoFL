import matplotlib as mpl
import matplotlib.pyplot as plt
import json, sys
sys.path.insert(1, '../')
import AGX.dvfs as agx
import TX2.dvfs as tx2
import numpy as np

Home_folder = '/Users/hongpengguo/Desktop/Spring22/Middleware22/figures'

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
# colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
markers = ['o','D','s','>','D','^']

Devices = ['TX2', 'AGX']
Tasks = ['CIFAR10', 'ImageNet', 'IMDB']
Ddls = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

TX2_Data = [[None for _ in range(6)] for _ in range(3)]
AGX_Data = [[None for _ in range(6)] for _ in range(3)]

Data = [TX2_Data, AGX_Data]


for i in range(len(Tasks)):
    for j in range (len(Ddls)):
        with open('../AGX/results/{}_{:.1f}.json'.format(Tasks[i], Ddls[j])) as f:
            AGX_Data[i][j] = json.load(f)
        with open('../TX2/results/{}_{:.1f}.json'.format(Tasks[i], Ddls[j])) as f:
            TX2_Data[i][j] = json.load(f)


def plot_energy_curve_by_round(device, task_name, max_ddl):
    d, t, m = Devices.index(device), Tasks.index(task_name), Ddls.index(max_ddl)
    baseline = Data[d][t][m]['Baseline'][:40]
    bofl = Data[d][t][m]['BoFL'][:40]
    oracle = Data[d][t][m]['Oracle'][:40]
    ddls = Data[d][t][m]['ddls'][:40]

    explore_start = Data[d][t][m]['phase'][0] + 1
    Bayesian_start = Data[d][t][m]['phase'][1] + 1
    exploit_start = Data[d][t][m]['phase'][2] + 1

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(9,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, sharex=True)
    x = np.arange(1, 41)
    ax1.plot(x, np.array(baseline), marker=markers[0], markersize=5,  label='Max',linestyle="--")
    ax1.plot(x, np.array(oracle), marker=markers[2], markersize=5,  label='Oracle', linestyle="-")
    ax1.plot(x, np.array(bofl), marker=markers[1], markersize=5,  label='BoFL', linestyle=":")
    ax1.axvspan(explore_start, Bayesian_start, alpha=0.2, color='red', lw=0, label='Phase 1')
    ax1.axvspan(Bayesian_start, exploit_start, alpha=0.2, color='green', lw=0, label='Phase 2')
    ax1.axvspan(exploit_start, 40, alpha=0.2, color='blue', lw=0, label='Phase 3')
    ax1.set_ylabel('Energy Consumed (J)', fontsize='16')
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:4.0f}'))

    ax2.bar(x, ddls, edgecolor='black')
    ax2.set_xlabel('Round Number', fontsize='16')
    ax2.set_ylim(Data[d][t][m]['min_ddl'], Data[d][t][m]['min_ddl'] * max_ddl)
    ax2.set_xlim(0, 41)
    ax2.set_ylabel('DDL (s)', fontsize='16')
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:4.0f}'))

    legend = fig.legend(ncol=3, fancybox=True, fontsize='15', loc='lower center',  bbox_to_anchor=(0., 0.865, 1.0, 1.0))
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()
    plt.savefig('{}/{}_{}_{:.1f}.pdf'.format(Home_folder, device, task_name, max_ddl), 
                                             dpi=600, pad_inches = 0, bbox_inche='tight')


def plot_pareto_graph(device, task_name, max_ddl):
    d, t, m = Devices.index(device), Tasks.index(task_name), Ddls.index(max_ddl)
    baseline = Data[d][t][m]['Baseline'][:40]
    bofl = Data[d][t][m]['BoFL'][:40]
    oracle = Data[d][t][m]['Oracle'][:40]
    ddls = Data[d][t][m]['ddls'][:40]

    # To  be finished
    

if __name__ == '__main__':
    plot_energy_curve_by_round('AGX', 'IMDB', 2.0)
    plot_energy_curve_by_round('AGX', 'CIFAR10', 2.0)
    plot_energy_curve_by_round('AGX', 'ImageNet', 2.0)


    plot_energy_curve_by_round('TX2', 'IMDB', 2.0)
    plot_energy_curve_by_round('TX2', 'CIFAR10', 2.0)
    plot_energy_curve_by_round('TX2', 'ImageNet', 2.0)

