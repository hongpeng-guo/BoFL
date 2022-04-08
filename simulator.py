import math, time, timeit
import itertools, random
import numpy as np
import dvfs, json
from getPareto import load_json, get_Pareto
import gpflow
import numpy as np
from collections import OrderedDict
import tensorflow as tf


class Simulation:

    def __init__(self, exp_name):

        if exp_name == 'CIFAR10':
            self.batch_size = 32
            self.data_size = 320
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
        self.explore_size = 20
        self.Bayesian_rounds = 5
        self.Bayesian_batch_size = 5
        self.tau = 5

        random.seed(3793)
        self.profile_res = self.generate_normalized_profile_results()
        self.ddls = self.generate_ddls()


    def generate_normalized_profile_results(self):

        profile_res = load_json(self.exp_name + '.json')
        res = {tuple([conf[i]/self.max_conf[i] for i in range(3)]): value for conf, value in profile_res.items()}
        return OrderedDict(sorted(res.items()))
        
    def generate_ddls(self):
        min_time = self.profile_res[(1.0, 1.0, 1.0)][0] * self.workload
        return [min_time * random.uniform(1.0, 2.0) for _ in range(self.rounds)]


    def RUN_ALGORITHM(self):

        self.round_counter = 0
        self.observations = []
        
        explore_ids = random.sample(range(len(self.profile_res) - 1), k=self.explore_size - 1)
        explore_ids = [len(self.profile_res) -1] + explore_ids
        print(explore_ids)
        self.init_explore_points = [list(self.profile_res)[i] for i in explore_ids]

        enery_res = []

        while len(self.observations) < len(self.init_explore_points):
            energy = self.run_explore_round()
            enery_res.append(energy)

        for _ in range(Bayesian_rounds):
            energy = self.run_Bayesian_round()
            energy_res.append(energy)

        while self.round_counter < self.rounds:
            energy = self.run_exploit_round()
            energy_res.append(energy)

        return energy_res


    def run_explore_round(self):

        min_batch_time = self.profile_res[self.init_explore_points[0]][0]
        min_time_energy = self.profile_res[self.init_explore_points[0]][1]

        energy, ddl, workload = 0, self.ddls[self.round_counter], self.workload
        
        while workload > 0 and (ddl - tau) / min_batch_time > workload:
            if self.observations == self.init_explore_points:
                break
            conf = self.init_explore_points[len(self.observations)]
            b_size = math.floor(self.tau / self.profile_res[conf][0])
            if workload >= b_size:
                self.observations.append(conf)
            else:
                b_size = workload
            energy += b_size * self.profile_res[conf][1]
            ddl -= bsize * self.profile_res[conf][0]
            workload -= b_size

        if workload > 0:
            if (ddl - tau) / min_batch_time < workload:
                energy += min_time_energy * workload
            else:
                energy += self.run_exploit(self.observations, workload, ddl)

        self.round_counter += 1
        return energy


    def run_exploit(self, observations, workload, ddl):

    
    def run_exploit_round(self):

        energy = self.run_exploit(self.observations, self.workload, self.ddls[self.round_counter])
        self.round_counter += 1



if __name__ == '__main__':
    s = Simulation('CIFAR10')
    # print(s.profile_res)
    s.run_algorithm()




