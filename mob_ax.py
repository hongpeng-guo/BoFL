# Need to run in the python3.9 environment
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

import torch, dvfs, json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from ax import *
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_NEHVI, get_MOO_PAREGO
from ax.modelbridge.modelbridge_utils import observed_hypervolume

from ax import optimize

confSpaceInt = dvfs.CONFIG_SPACE

x1 = ChoiceParameter(name="x1", parameter_type=ParameterType.INT, values=confSpaceInt[0])
x2 = ChoiceParameter(name="x2", parameter_type=ParameterType.INT, values=confSpaceInt[1])
x3 = ChoiceParameter(name="x3", parameter_type=ParameterType.INT, values=confSpaceInt[2])

search_space = SearchSpace(parameters=[x1, x2, x3])

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}

function_data = load_json('CIFAR10.json')

def evaluate(conf):
    # print(evaluation_dict, conf in evaluation_dict)
    conf = tuple(conf.numpy().tolist())
    t, e = function_data[conf]
    return (t, e)

class MetricTime(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return float(evaluate(torch.tensor(x))[0])
    
class MetricEnergy(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return float(evaluate(torch.tensor(x))[1])


metric_time = MetricTime("time", ["x1", "x2", "x3"], noise_sd=0.0, lower_is_better=True)
metric_energy = MetricEnergy("energy", ["x1", "x2", "x3"], noise_sd=0.0, lower_is_better=True)

mo = MultiObjective(objectives=[Objective(metric=metric_time), Objective(metric=metric_energy)])

optimization_config = MultiObjectiveOptimizationConfig(objective=mo)

N_INIT = 20
N_BATCH = 30

def build_experiment():
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment

def initialize_experiment(experiment):
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)

    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()

    return experiment.fetch_data()

ehvi_experiment = build_experiment()
ehvi_data = initialize_experiment(ehvi_experiment)

from time import time

ehvi_hv_list = []
ehvi_model = None
for i in range(N_BATCH):   
    ehvi_model = get_MOO_EHVI(
        experiment=ehvi_experiment, 
        data=ehvi_data,
    )
    t0 = time()
    generator_run = ehvi_model.gen(1)
    # generator_run = ehvi_model.gen(3)
    t1 = time()
    print(i, t1 - t0)
    trial = ehvi_experiment.new_trial(generator_run=generator_run)
    # trial = ehvi_experiment.new_batch_trial(generator_run=generator_run)
    print('======', trial.arms, '======')
    trial.run()
    ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
    t2 = time()
    
    print(t1-t0, t2-t1)
    
    exp_df = exp_to_df(ehvi_experiment)
    outcomes = np.array(exp_df[['time', 'energy']], dtype=np.double)
    try:
        hv = observed_hypervolume(modelbridge=ehvi_model)
    except:
        hv = 0
        print("Failed to compute hv")
    ehvi_hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[['time', 'energy']], dtype=np.double)

print(ehvi_outcomes)
