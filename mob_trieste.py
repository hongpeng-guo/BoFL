import math, time, timeit
import itertools
import numpy as np
import dvfs, json

import gpflow
import numpy as np
import tensorflow as tf

import trieste
from trieste.acquisition.function import ExpectedHypervolumeImprovement, Fantasizer
from trieste.acquisition import LocalPenalization
from trieste.acquisition.function.multi_objective import BatchMonteCarloExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization, DiscreteThompsonSampling
from trieste.data import Dataset
from trieste.models import TrainableModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.models.interfaces import ReparametrizationSampler
from trieste.space import DiscreteSearchSpace
from trieste.objectives import multi_objectives
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.acquisition.multi_objective.pareto import (
    Pareto,
    get_reference_point,
)
from trieste.acquisition.multi_objective.partition import prepare_default_non_dominated_partition_bounds

np.random.seed(3793)
tf.random.set_seed(3793)

confSpaceInt = dvfs.CONFIG_SPACE
num_dim = len(confSpaceInt)
print(num_dim)
num_objective = 2


confSpaceMax = [max(eachDim) for eachDim in confSpaceInt]
confSpaceFloat = [[d / confSpaceMax[i] for d in confSpaceInt[i]] for i in range(num_dim)]

space_min = [float(eachDim[0]) for eachDim in confSpaceFloat]
space_max = [float(eachDim[-1]) for eachDim in confSpaceFloat]

search_space = [list(each) for each in list(itertools.product(*confSpaceFloat))]
search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))

confSpaceMax = tf.constant(confSpaceMax, dtype=tf.float64)

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return {tuple([int(i) for i in key.split(',')]): tuple(value) for key, value in data.items()}

function_data = load_json('ImageNet.json')

def getSysProfileResult(conf):
    tf.debugging.assert_shapes([(conf, (..., 3))], message="only allow 2d input")
    res = []
    conf = tf.cast(tf.multiply(conf, confSpaceMax), dtype=tf.int32)
    for i in range(conf.shape[0]):
        print('step {:d}'.format(i))
        print(conf[i].numpy().tolist())
        t, e = function_data[tuple(conf[i].numpy().tolist())]
        res.append([t, e])
    res = tf.stack(res, axis=0)
    return tf.cast(res, dtype=tf.float64)

class SysConfProblem(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):
    bounds = [space_min, space_max]
    dim = num_dim

    def objective(self):
        return getSysProfileResult

    # This one is fake
    def gen_pareto_optimal_points(self, n: int, seed: int):
        return getSysProfileResult(search_space[0])

sysconf = SysConfProblem().objective()
observer = trieste.objectives.utils.mk_observer(sysconf)

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

print(initial_data.observations)

# plot_mobo_points_in_obj_space(initial_data.observations, \
#     xlabel="Runming time (second/image)", ylabel="Energy efficientcy (joule/image)")
# plt.savefig('init_observs.jpg')

def build_stacked_independent_objectives_model(
    data: Dataset, num_output
) -> TrainableModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-5)
        gprs.append((GaussianProcessRegression(gpr), 1))

    return TrainableModelStack(*gprs)

model = build_stacked_independent_objectives_model(initial_data, num_objective)

# ehvi = ExpectedHypervolumeImprovement()
# rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=ehvi)

fant_ehvi = Fantasizer(ExpectedHypervolumeImprovement())
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(
    builder=fant_ehvi, num_query_points=1
)

num_steps = 30

dataset = initial_data
ref_point = get_reference_point(dataset.observations)
print(ref_point)
print('Initial HV value:', Pareto(dataset.observations).hypervolume_indicator(ref_point))

model.update(dataset)
model.optimize(dataset)

for step in range(num_steps):

    print("Asking for new point to observe")
    t0 = timeit.default_timer()
    ask_tell = AskTellOptimizer(search_space, dataset, model, acquisition_rule=rule, fit_model=False)
    t1 = timeit.default_timer()
    new_point = ask_tell.ask()
    t2 = timeit.default_timer()

    print(new_point)

    new_data_point = observer(new_point)

    print(new_data_point.observations.numpy())
    
    t3 = timeit.default_timer()
    dataset = dataset + new_data_point

    print('HV value:', Pareto(dataset.observations).hypervolume_indicator(ref_point))

    print("Training models externally")
    model.update(dataset)
    t4 = timeit.default_timer()
    model.optimize(dataset)
    t5 = timeit.default_timer()

    print('Build: {:6.2f}; ASK: {:6.2f}; OB: {:6.2f}; UP: {:6.2f}; OP :{:6.2f}'.format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4))

data_query_points = dataset.query_points
data_observations = dataset.observations

print(data_query_points)
print(data_observations)

unique_points = set([tuple(x) for x in data_query_points[20:].numpy().tolist()])
print(len(unique_points))