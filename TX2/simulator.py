import math, time, timeit
import itertools, random
import numpy as np
import dvfs, json
from exploitOpt import load_json, getParetos, exploitOpt
import gpflow
import numpy as np
from collections import OrderedDict

import tensorflow as tf
from trieste.acquisition.function import ExpectedHypervolumeImprovement, Fantasizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.models import TrainableModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import DiscreteSearchSpace
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point


import powerLog as pl
import sys
from powerLog import PowerLogger

SPACE_SIZE = len(dvfs.CPU_FREQ_TABLE) * len(dvfs.GPU_FREQ_TABLE) * len(dvfs.EMC_FREQ_TABLE)


class BayesianOpt:

    def __init__(self, search_space, batch_size):

        self.num_objectives = 2
        self.search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))
        
        fant_ehvi = Fantasizer(ExpectedHypervolumeImprovement())
        self.rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=fant_ehvi, num_query_points=batch_size)


    def build_stacked_independent_objectives_model(self, data: Dataset) -> TrainableModelStack:
        gprs = []
        for idx in range(self.num_objectives):
            single_obj_data = Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            gpr = build_gpr(single_obj_data, self.search_space, likelihood_variance=1e-5)
            gprs.append((GaussianProcessRegression(gpr), 1))

        self.model = TrainableModelStack(*gprs)

    def ask_for_suggestions(self, dataset):

        ask_tell = AskTellOptimizer(self.search_space, dataset, self.model, acquisition_rule=self.rule, fit_model=True)
        return ask_tell.ask()


class Simulation:

    def __init__(self, exp_name, ddl_max):

        if exp_name == 'CIFAR10':
            self.batch_size = 32
            self.data_size = 480
            self.epoch_size = 5
        elif exp_name == 'ImageNet':
            self.batch_size = 8
            self.data_size = 240
            self.epoch_size = 2
        elif exp_name == 'IMDB':
            self.batch_size = 8
            self.data_size = 160
            self.epoch_size = 4

        self.exp_name = exp_name
        self.workload = self.data_size / self.batch_size * self.epoch_size
        self.max_conf = tuple([dim[-1] for dim in dvfs.CONFIG_SPACE])

        self.rounds = 100
        self.explore_size = math.ceil(SPACE_SIZE * 0.01)
        self.Bayesian_observation_threshold = math.ceil(SPACE_SIZE * 0.03)
        self.Bayesian_batch_size = None
        self.HV_threshold = 0.01
        self.tau = 5

        self.profile_res = self.generate_normalized_profile_results()
        self.ddls = self.generate_ddls(ddl_max)


    def generate_Bayesian_batch_size(self):

        prev_suggestion = math.floor(sum(self.ddls[:self.round_counter]) / self.round_counter / self.tau)
        self.Bayesian_batch_size = min(prev_suggestion, 10)

    def generate_normalized_profile_results(self):

        profile_res = load_json(self.exp_name + '.json')
        res = {tuple([conf[i]/self.max_conf[i] for i in range(3)]): value for conf, value in profile_res.items() if conf[0] in dvfs.CPU_FREQ_TABLE}
        return OrderedDict(sorted(res.items()))
        
    def generate_ddls(self, ddl_max):
        min_time = self.profile_res[(1.0, 1.0, 1.0)][0] * self.workload
        return [min_time * random.uniform(1.0, ddl_max) for _ in range(self.rounds)]


    def RUN_ALGORITHM(self):

        self.round_counter = 0
        self.observations = []

        phase_starter = []
        ob_round_counter = []

        logger = PowerLogger(interval=0.1, nodes=list(filter(lambda n: n[0].startswith('module/'), pl.getNodes())))
        
        explore_ids = random.sample(range(len(self.profile_res) - 1), k=self.explore_size - 1)
        explore_ids = [len(self.profile_res) -1] + explore_ids
        print(explore_ids)
        self.init_explore_points = [list(self.profile_res)[i] for i in explore_ids]

        energy_res = []

        phase_starter.append(self.round_counter)
        while len(self.observations) < len(self.init_explore_points):
            energy = self.run_explore_round()
            energy_res.append(energy)
            ob_round_counter.append(len(self.observations))

        # mob = BayesianOpt([list(k) for k in self.profile_res.keys()], self.Bayesian_batch_size)
        # mob.build_stacked_independent_objectives_model(self.build_trieste_dataset())
        # suggestions = mob.ask_for_suggestions(self.build_trieste_dataset())

        HV_last = 0
        ref_point = get_reference_point(self.build_trieste_dataset().observations)
        self.generate_Bayesian_batch_size()
        mob = BayesianOpt([list(k) for k in self.profile_res.keys()], self.Bayesian_batch_size)
        self.run_Bayesian_warmup(mob, explore_ids)


        phase_starter.append(self.round_counter)
        Bayesian_overhead = []
        while True:

            dataset = self.build_trieste_dataset()
            HV_new = Pareto(dataset.observations).hypervolume_indicator(ref_point)
            HV_improvement = (HV_new - HV_last) / HV_last
            print("EHVI improvement is: {}".format(HV_improvement))
            HV_last = HV_new

            print(len(dataset.observations))
            if len(dataset.observations) > self.Bayesian_observation_threshold and HV_improvement < 0.01:
                break

            Bayesian_start = time.time()
            logger.start()
            suggestions = mob.ask_for_suggestions(dataset)
            logger.stop()
            Bayesian_energy = logger.getTotalEnergy() / 1000
            Bayesian_end = time.time()
            Bayesian_overhead.append((Bayesian_end - Bayesian_start, Bayesian_energy))
            # energy_res[-1] += Bayesian_energy
            logger.reset()
            print('bayesian time cost is {}s.'.format(Bayesian_end - Bayesian_start))

            energy = self.run_Bayesian_round(suggestions)
            energy_res.append(energy)
            ob_round_counter.append(len(self.observations))

        phase_starter.append(self.round_counter)
        while self.round_counter < self.rounds:
            energy = self.run_exploit_round(self.observations)
            energy_res.append(energy)

        return energy_res, phase_starter, Bayesian_overhead, ob_round_counter

    def build_trieste_dataset(self):
        conf_mat = tf.constant([list(i) for i in self.observations], dtype=tf.float64)
        print(conf_mat)
        value_mat = tf.constant([self.profile_res[k] for k in self.observations], dtype=tf.float64)
        print(value_mat)
        return Dataset(conf_mat, value_mat)

    def run_Bayesian_warmup(self, Bayesian_object, warmup_ids):
        keys = [list(self.profile_res.keys())[i] for i in warmup_ids]
        warmup_confs = [list(k) for k in keys]
        warmup_values = [self.profile_res[k] for k in keys]
        conf_mat = tf.constant(warmup_confs, dtype=tf.float64)
        value_mat = tf.constant(warmup_values, dtype=tf.float64)
        warmup_dataset = Dataset(conf_mat, value_mat)
        Bayesian_object.build_stacked_independent_objectives_model(warmup_dataset)
        Bayesian_object.ask_for_suggestions(warmup_dataset)
        print('Finished system warmup')


    def run_explore_round(self):

        min_batch_time = self.profile_res[self.init_explore_points[0]][0]
        min_time_energy = self.profile_res[self.init_explore_points[0]][1]

        energy, ddl, workload = 0, self.ddls[self.round_counter], self.workload
        
        while workload > 0 and (ddl - self.tau) / min_batch_time > workload:
            if self.observations == self.init_explore_points:
                break
            conf = self.init_explore_points[len(self.observations)]
            bsize = math.floor(self.tau / self.profile_res[conf][0])
            if workload >= bsize:
                self.observations.append(conf)
            else:
                bsize = workload
            energy += bsize * self.profile_res[conf][1]
            ddl -= bsize * self.profile_res[conf][0]
            workload -= bsize

        if workload > 0:
            if (ddl - self.tau) / min_batch_time < workload:
                energy += min_time_energy * workload
            else:
                energy += self.run_exploit(self.observations, workload, ddl)

        self.round_counter += 1
        return energy


    def run_Bayesian_round(self, suggestions):

        suggestions = [tuple(suggestions[i].numpy().tolist()) for i in range(suggestions.shape[0])]
        round_observations = []

        min_batch_time = self.profile_res[self.init_explore_points[0]][0]
        min_time_energy = self.profile_res[self.init_explore_points[0]][1]

        energy, ddl, workload = 0, self.ddls[self.round_counter], self.workload
        
        while workload > 0 and (ddl - self.tau) / min_batch_time > workload:
            if round_observations == suggestions:
                break
            conf = suggestions[len(round_observations)]
            bsize = math.floor(self.tau / self.profile_res[conf][0])
            if workload >= bsize:
                round_observations.append(conf)
            else:
                bsize = workload
            energy += bsize * self.profile_res[conf][1]
            ddl -= bsize * self.profile_res[conf][0]
            workload -= bsize

        self.observations.extend(round_observations)

        if workload > 0:
            if (ddl - self.tau) / min_batch_time < workload:
                energy += min_time_energy * workload
            else:
                energy += self.run_exploit(self.observations, workload, ddl)

        self.round_counter += 1
        return energy


    def run_exploit(self, observations, workload, ddl):

        ob_values = [self.profile_res[k] for k in observations]
        return exploitOpt(ob_values, workload, ddl)


    
    def run_exploit_round(self, observations):

        energy = self.run_exploit(observations, self.workload, self.ddls[self.round_counter])
        self.round_counter += 1
        return energy


    def RUN_BASELINE(self):

        min_time_energy = self.profile_res[(1.0, 1.0, 1.0)][1]
        energy_res = []
        for _ in range(self.rounds):
            energy_res.append(self.epoch_size * self.data_size / self.batch_size * min_time_energy)
        return energy_res

    
    def RUN_OPTIMAL(self):

        self.round_counter = 0
        observations = self.profile_res.keys()

        energy_res = []
        for _ in range(self.rounds):
            energy_res.append(self.run_exploit_round(observations))
        return energy_res

        



if __name__ == '__main__':

    seed_dict = {'CIFAR10': 3793, 'ImageNet': 3456, 'IMDB': 4567}
    exp, ddl_max = sys.argv[1], float(sys.argv[2])
    random.seed(seed_dict[exp])
    s = Simulation(exp, ddl_max)
    A_res, phase_starter, BO_overhead, ob_round_counter = s.RUN_ALGORITHM()
    B_res = s.RUN_BASELINE()
    O_res = s.RUN_OPTIMAL()

    print(A_res, sum(A_res))
    print(B_res, sum(B_res))
    print(O_res, sum(O_res))

    print(phase_starter, BO_overhead)

    RESULTS = {
        'Baseline': B_res,
        'Oracle': O_res,
        'BoFL': A_res,
        'observations': [tuple([round(ob[i] * s.max_conf[i] / 100) * 100 for i in range(3)]) for ob in s.observations],
        'ob_round_counter': ob_round_counter,
        'phase': phase_starter,
        'overhead': BO_overhead,
        'ddls': s.ddls,
        'min_ddl': s.profile_res[(1.0, 1.0, 1.0)][0] * s.workload,
    }

    file_name = "results/{}_{}.json".format(exp, sys.argv[2])
    with open(file_name , 'w') as fp:
        json.dump(RESULTS, fp)
        
    # import matplotlib.pyplot as plt
    
    # plt.figure(figsize=(8,3))
    # x = np.arange(1, 101)
    # plt.plot(x, np.array(B_res), color='blue', label='baseline')
    # plt.plot(x, np.array(A_res), color='red', label='our algorithm')
    # plt.plot(x, np.array(O_res), color='green', label='oracle')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=3,)
    # plt.xlabel('Round Number')
    # plt.ylabel('Energy Consumed (J)')
    # plt.savefig('{}_sim.jpg'.format(exp))




