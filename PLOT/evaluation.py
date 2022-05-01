from email.mime import base
import matplotlib as mpl
import matplotlib.pyplot as plt
import json, sys
sys.path.insert(1, '../')
import AGX.dvfs as agx
import TX2.dvfs as tx2
import numpy as np

Home_folder = '/Users/hongpengguo/Desktop/Spring22/Middleware22/figures'

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 4
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
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
    ax1.plot(x, np.array(baseline), marker=markers[0], markersize=6,  label='Performance',linestyle="--", color=colors[0])
    ax1.plot(x, np.array(oracle), marker=markers[2], markersize=6,  label='Oracle', linestyle="-", color=colors[1])
    ax1.plot(x, np.array(bofl), marker=markers[1], markersize=6,  label='BoFL', linestyle=":", color=colors[2])
    ax1.axvspan(explore_start-0.5, Bayesian_start-0.5, alpha=0.2, color='red', lw=0, label='Phase 1')
    ax1.axvspan(Bayesian_start-0.5, exploit_start-0.5, alpha=0.2, color='green', lw=0, label='Phase 2')
    ax1.axvspan(exploit_start-0.5, 40, alpha=0.2, color='blue', lw=0, label='Phase 3')
    # ax1.fill_between(x, np.array(baseline), np.array(bofl), hatch = '///', alpha=0)
    # ax1.fill_between(x, np.array(bofl), np.array(oracle), hatch = '*', alpha=0)
    ax1.set_ylabel('Energy Consumed (J)', fontsize='18')
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:4.0f}'))
    # ax1.grid(axis='y', linestyle='--', linewidth=1)

    ax2.bar(x, ddls, edgecolor='black')
    ax2.set_xlabel('Round Number', fontsize='18')
    ax2.set_ylim(Data[d][t][m]['min_ddl'], Data[d][t][m]['min_ddl'] * max_ddl)
    ax2.set_xlim(0, 41)
    ax2.set_ylabel('DDL (s)', fontsize='18')
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:4.0f}'))
    # ax2.grid(axis='y', linestyle='--', linewidth=1)

    legend = fig.legend(ncol=3, fancybox=True, fontsize='16', loc='lower center',  bbox_to_anchor=(0., 0.86, 1.0, 1.0))
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()
    plt.savefig('{}/{}_{}_{:.1f}_energy.pdf'.format(Home_folder, device, task_name, max_ddl), 
                                             dpi=600, pad_inches = 0, bbox_inche='tight')

def get_Pareto(performace):

    sorted_metrics = sorted(performace)
    pareto = [sorted_metrics[0]]
    for i in range(1, len(sorted_metrics)):
        if sorted_metrics[i][1] < pareto[-1][1]:
            pareto.append(sorted_metrics[i])
    return pareto

def plot_pareto_graph(device, task_name, max_ddl):
    d, t, m = Devices.index(device), Tasks.index(task_name), Ddls.index(max_ddl)
    obsevations = Data[d][t][m]['observations']
    ob_round_counter = Data[d][t][m]['ob_round_counter']
    phase = Data[d][t][m]['phase']

    if device == 'AGX':
        cpu_configs = agx.CPU_FREQ_TABLE
    else:
        cpu_configs = tx2.CPU_FREQ_TABLE

    with open('../{}/{}.json'.format(device, task_name)) as f:
        data = json.load(f)
    all_points = [list(value) for key, value in data.items() if int(key.split(',')[0]) in cpu_configs]
    
    real_pareto = get_Pareto(all_points)

    tried_points = [list(data[','.join([str(i) for i in ob])]) for ob in obsevations]

    tried_pareto = get_Pareto(tried_points)

    all_points = np.array(all_points)
    real_pareto = np.array(real_pareto)
    tried_points = np.array(tried_points)
    tried_pareto = np.array(tried_pareto)

    xmin, xmax = 0.95 * np.min(real_pareto[:,0]), 1.2 * np.max(real_pareto[:,0])
    ymin, ymax = 0.98 * np.min(real_pareto[:,1]), 1.2 * np.max(real_pareto[:,1])

    fig, ax = plt.subplots(figsize=(9,6))   
    ax.scatter(all_points[:, 0], all_points[:, 1], marker='.',  s=120, lw=0, color="grey", alpha=0.4, label='Other Configurations')
    ax.scatter(tried_points[:, 0], tried_points[:, 1], marker='o', s=120, lw=0,  alpha=0.5, label='BoFL Explorations')
    ax.scatter(tried_pareto[:, 0], tried_pareto[:, 1], marker='s', s=120, lw=0, color="blue", alpha=0.8, label='BoFL Perato Front')
    ax.scatter(real_pareto[:, 0], real_pareto[:, 1], marker='*', s=120, lw=0,  color="red", alpha=0.8, label='Global Perato Front')
    ax.set_ylabel('Energy Consumption per minibatch (J)', fontsize='18')
    ax.set_xlabel('Execution Latency per minibatch (s)', fontsize='18')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.2f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.1f}'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.text(1.03 * xmin, 1.03 *ymin, "Better", ha="left", va="bottom", rotation=45, size=18,
            bbox=dict(boxstyle="larrow,pad=0.3", fc="cyan", ec="black", lw=2))

    legend = fig.legend(ncol=1, fancybox=True, fontsize='18', loc='lower left',  bbox_to_anchor=(0.1, 0.7, 1.0, 1.0))
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()
    plt.savefig('{}/{}_{}_{:.1f}_pareto.pdf'.format(Home_folder, device, task_name, max_ddl), 
                                             dpi=600, pad_inches = 0, bbox_inche='tight')


def plot_energy_w_ddl_max():
    
    def get_regret_overhead(device, task_name, max_ddl):
        d, t, m = Devices.index(device), Tasks.index(task_name), Ddls.index(max_ddl)
        baseline = sum(Data[d][t][m]['Baseline'])
        bofl = sum(Data[d][t][m]['BoFL'])
        oracle = sum(Data[d][t][m]['Oracle'])
        return bofl/oracle - 1, 1 - bofl / baseline

    Plot_data = [[[None for _ in range(6)] for _ in range(2)] for _ in range(3)]

    for i, task_name in enumerate(Tasks):
        for j, device in enumerate(['AGX']):
            for k, max_ddl in enumerate(Ddls):
                regret, overhead = get_regret_overhead(device, task_name, max_ddl)
                Plot_data[i][0][k] = regret
                Plot_data[i][1][k] = overhead

    Plot_data = np.array(Plot_data)

    fig, axs = plt.subplots(3,2,figsize=(12,7), gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.2}, sharex=True)
    
    width = 0.4

    x = np.arange(6)
    axs[0][0].set_title('(a) CIFAR10-ViT', fontsize='14', fontweight='bold')
    axs[0][0].set_xticks(x)
    axs[0][0].bar(x , 100 * Plot_data[0][0], width, edgecolor='black', color=colors[0])
    axs[0][0].set_xticklabels(['{:1.1f}x'.format(ddl) for ddl in Ddls])
    axs[0][0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:1.0f}'))
    axs[0][0].yaxis.grid(linestyle='--',  linewidth=1)

    axs[1][0].set_title('(c) ImageNet-ResNet50', fontsize='14', fontweight='bold')
    axs[1][0].set_xticks(x)
    axs[1][0].bar(x , 100 * Plot_data[0][0], width, edgecolor='black', color=colors[0])
    axs[1][0].set_xticklabels(['{:1.1f}x'.format(ddl) for ddl in Ddls])
    axs[1][0].set_ylabel('Regret Compared to Oracle (%)', fontsize='18')
    axs[1][0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:1.0f}'))
    axs[1][0].yaxis.grid(linestyle='--',  linewidth=1)

    axs[2][0].set_title('(e) IMDB-LSTM', fontsize='14', fontweight='bold')
    axs[2][0].set_xticklabels(['{:1.1f}x'.format(ddl) for ddl in Ddls])
    axs[2][0].bar(x , 100 * Plot_data[0][0], width, edgecolor='black', color=colors[0])
    axs[2][0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:1.0f}'))
    axs[2][0].yaxis.grid(linestyle='--',  linewidth=1)


    axs[0][1].set_title('(b) CIFAR10-ViT', fontsize='14', fontweight='bold')
    axs[0][1].set_xticklabels(['{:1.1f}x'.format(ddl) for ddl in Ddls])
    axs[0][1].yaxis.tick_right()
    axs[0][1].bar(x , 100 * Plot_data[0][1], width, edgecolor='black', color=colors[1], hatch='/')
    axs[0][1].set_xticklabels(Ddls)
    axs[0][1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.0f}'))
    axs[0][1].yaxis.grid(linestyle='--',  linewidth=1)

    axs[1][1].set_title('(d) ImageNet-ResNet50', fontsize='14', fontweight='bold')
    axs[1][1].set_xticklabels(['{:1.1f}x'.format(ddl) for ddl in Ddls])
    axs[1][1].yaxis.tick_right()
    axs[1][1].bar(x , 100 * Plot_data[0][1], width, edgecolor='black', color=colors[1], hatch='/')
    axs[1][1].set_xticklabels(Ddls)
    axs[1][1].set_ylabel('Improvement Compared to Performance (%)', fontsize='18')
    axs[1][1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.0f}'))
    axs[1][1].yaxis.grid(linestyle='--',  linewidth=1)

    axs[2][1].set_title('(f) IMDB-LSTM', fontsize='14', fontweight='bold')
    axs[2][1].set_xticklabels(['{:1.1f}x'.format(ddl) for ddl in Ddls])
    axs[2][1].yaxis.tick_right()
    axs[2][1].bar(x , 100 * Plot_data[0][1], width, edgecolor='black', color=colors[1], hatch='/')
    axs[2][1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.0f}'))
    axs[2][1].yaxis.grid(linestyle='--',  linewidth=1)

    fig.text(0.5, 0.02, 'Normalized Maximun Deadline Length', va='center', ha='center', fontsize=18)

    plt.tight_layout()
    plt.savefig('{}/energy_w_ddls.pdf'.format(Home_folder), dpi=600, pad_inches = 0, bbox_inche='tight')


def plot_Bayesain_overhead_per_round():
    TX2_latency = np.array(Data[0][0][1]['overhead'])[:, 0]
    TX2_energy = np.array(Data[0][0][1]['overhead'])[:, 1]

    AGX_latency = np.array(Data[1][0][1]['overhead'])[:, 0]
    AGX_energy = np.array(Data[1][0][1]['overhead'])[:, 1]

    TX2_latency_mean = np.mean(TX2_latency)
    TX2_latency_error = np.abs(np.percentile(TX2_latency, [25, 75]) - TX2_latency_mean).reshape(2,1)
    TX2_energy_mean = np.mean(TX2_energy)
    TX2_energy_error = np.abs(np.percentile(TX2_energy, [25, 75]) - TX2_energy_mean).reshape(2,1)

    AGX_latency_mean = np.mean(AGX_latency)
    AGX_latency_error = np.abs(np.percentile(AGX_latency, [25, 75]) - AGX_latency_mean).reshape(2,1)
    AGX_energy_mean = np.mean(AGX_energy)
    AGX_energy_error = np.abs(np.percentile(AGX_energy, [25, 75]) - AGX_energy_mean).reshape(2,1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,6),gridspec_kw={'hspace': 0.5})
    labels = ['TX2', 'AGX']
    x = np.arange(len(labels))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.bar(0, TX2_latency_mean, yerr=TX2_latency_error, width=0.3, hatch=None, edgecolor='black', color=colors[3])
    ax1.bar(1, AGX_latency_mean, yerr=AGX_latency_error, width=0.3, hatch='/', edgecolor='black', color=colors[4])
    ax1.yaxis.grid(linestyle='--',  linewidth=1.5)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_ylabel('Execution Latency (s)', fontsize='18')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.yaxis.tick_right()
    ax2.bar(0 + 0.05, TX2_energy_mean, yerr=TX2_energy_error, width=0.3, hatch=None, edgecolor='black', color=colors[3])
    ax2.bar(1 - 0.05, AGX_energy_mean, yerr=AGX_energy_error, width=0.3, hatch='/', edgecolor='black', color=colors[4])
    ax2.yaxis.grid(linestyle='--',  linewidth=1.5)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.set_ylabel('Energy Consumed (J)', fontsize='18')

    plt.tight_layout()
    plt.savefig('{}/Bayesian_overhead_value.pdf'.format(Home_folder), dpi=600, pad_inches = 0, bbox_inche='tight')


def plot_Bayesain_overhead_percentail():
    TX2_data, AGX_data = [], []
    for i in range(len(Tasks)):
        TX2_Bayesian_energy = np.sum(np.array(Data[0][i][1]['overhead'])[:, 1])
        AGX_Bayesian_energy = np.sum(np.array(Data[1][i][1]['overhead'])[:, 1])
        TX2_BoFL_energy = np.sum(np.array(Data[0][i][1]['BoFL']))
        AGX_BoFL_energy = np.sum(np.array(Data[1][i][1]['BoFL']))
        TX2_data.append(TX2_Bayesian_energy / TX2_BoFL_energy)
        AGX_data.append(AGX_Bayesian_energy / AGX_BoFL_energy)

    TX2_data = 100 * np.array(TX2_data)
    AGX_data = 100 * np.array(AGX_data)

    width, delta = 0.3, 0.01
    fig, ax = plt.subplots(figsize=(7,6))
    labels = ['CIFAR10', 'ImageNet', 'IMDB']
    x = np.arange(len(labels))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.bar(x - 0.5 * width - delta, TX2_data,  width=width, hatch=None, edgecolor='black', color=colors[3], label='TX2')
    ax.bar(x + 0.5 * width + delta, AGX_data,  width=width, hatch='/', edgecolor='black', color=colors[4], label='AGX')
    ax.yaxis.grid(linestyle='--',  linewidth=1.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:1.1f}'))
    ax.set_ylabel('Energy Overhead Generated by MBO (%)', fontsize='18')

    legend = fig.legend(ncol=1, fancybox=True, fontsize='18')
    legend.get_frame().set_edgecolor('black')
    plt.tight_layout()
    plt.savefig('{}/Bayesian_overhead_percentail.pdf'.format(Home_folder), dpi=600, pad_inches = 0, bbox_inche='tight')

if __name__ == '__main__':

    plot_energy_curve_by_round('AGX', 'IMDB', 2.0)
    plot_energy_curve_by_round('AGX', 'CIFAR10', 2.0)
    plot_energy_curve_by_round('AGX', 'ImageNet', 2.0)
    plot_energy_curve_by_round('TX2', 'IMDB', 2.0)
    plot_energy_curve_by_round('TX2', 'CIFAR10', 2.0)
    plot_energy_curve_by_round('TX2', 'ImageNet', 2.0)

    plot_pareto_graph('AGX', 'IMDB', 2.0)
    plot_pareto_graph('AGX', 'CIFAR10', 2.0)
    plot_pareto_graph('AGX', 'ImageNet', 2.0)
    plot_pareto_graph('TX2', 'IMDB', 2.0)
    plot_pareto_graph('TX2', 'CIFAR10', 2.0)
    plot_pareto_graph('TX2', 'ImageNet', 2.0)

    plot_energy_w_ddl_max()
    plot_Bayesain_overhead_per_round()
    plot_Bayesain_overhead_percentail()
