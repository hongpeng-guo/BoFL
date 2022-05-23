import json, sys
sys.path.insert(1, '../')
import AGX.dvfs as agx
import TX2.dvfs as tx2
import numpy as np

Home_folder = '/Users/hongpengguo/Desktop/Spring22/Middleware22/figures'

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}


TX2_CIFAR10_data = json.load(open('../TX2/results/CIFAR10_2.0.json'))
TX2_ImageNet_data = json.load(open('../TX2/results/ImageNet_2.0.json'))
TX2_IMDB_data = json.load(open('../TX2/results/IMDB_2.0.json'))
AGX_CIFAR10_data = json.load(open('../AGX/results/CIFAR10_2.0.json'))
AGX_ImageNet_data = json.load(open('../AGX/results/ImageNet_2.0.json'))
AGX_IMDB_data = json.load(open('../AGX/results/IMDB_2.0.json'))

pro_TX2_CIFAR10_data = load_json('../TX2/CIFAR10.json')
pro_TX2_ImageNet_data = load_json('../TX2/ImageNet.json')
pro_TX2_IMDB_data = load_json('../TX2/IMDB.json')
pro_AGX_CIFAR10_data = load_json('../AGX/CIFAR10.json')
pro_AGX_ImageNet_data = load_json('../AGX/ImageNet.json')
pro_AGX_IMDB_data = load_json('../AGX/IMDB.json')

def get_Pareto(performace):

    sorted_metrics = sorted(performace)
    pareto = [sorted_metrics[0]]
    for i in range(1, len(sorted_metrics)):
        if sorted_metrics[i][1] < pareto[-1][1]:
            pareto.append(sorted_metrics[i])
    return pareto

def get_explore_pareto_per_step(exp_data, profile_data):

    Result = []

    observations = exp_data['observations']
    performance = [profile_data[tuple(ob)] for ob in observations]
    step_count = [0] + exp_data['ob_round_counter']
    phase = [1] * exp_data['phase'][1] + [2] * (exp_data['phase'][2] - exp_data['phase'][1])

    pareto_performance = get_Pareto(performance)

    for i in range(1, len(step_count)):
        explore_count = step_count[i] - step_count[i-1]
        pareto_count = 0
        for j in range(step_count[i-1], step_count[i]):
            if performance[j] in pareto_performance:
                pareto_count += 1
        Result.append([explore_count, pareto_count, phase[i-1]])
    
    Result = np.array(Result)

    print(Result)

    return Result



if __name__ == '__main__':
    # get_explore_pareto_per_step(TX2_CIFAR10_data, pro_TX2_CIFAR10_data)
    # get_explore_pareto_per_step(TX2_ImageNet_data, pro_TX2_ImageNet_data)
    # get_explore_pareto_per_step(TX2_IMDB_data, pro_TX2_IMDB_data)
    
    get_explore_pareto_per_step(AGX_CIFAR10_data, pro_AGX_CIFAR10_data)
    get_explore_pareto_per_step(AGX_ImageNet_data, pro_AGX_ImageNet_data)
    get_explore_pareto_per_step(AGX_IMDB_data, pro_AGX_IMDB_data)

