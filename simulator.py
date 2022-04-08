import math, time, timeit
import itertools, random
import numpy as np
import dvfs, json
from getPareto import load_json, get_Pareto
import gpflow
import numpy as np
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
        self.max_conf = tuple([dim[-1] for dim in dvfs.CONFIG_SPACE])

        self.rounds = 100
        self.explore_size = 20
        self.Bayesian_batch_size = 5

        self.profile_res = self.generate_normalized_profile_results()
        self.ddls = self.generate_ddls()


    def generate_normalized_profile_results(self):

        profile_res = load_json(self.exp_name + '.json')
        return {tuple([conf[i]/self.max_conf[i] for i in range(3)]): value for conf, value in profile_res.items()}

        
    def generate_ddls(self):
        random.seed(3793)
        min_time = self.profile_res[(1.0, 1.0, 1.0)][0] * self.data_size / self.batch_size * self.epoch_size
        return [min_time * random.uniform(1.0, 2.0) for _ in range(self.rounds)]


    def run_explore_round(self):

        ddl = self.ddls[self.round_counter]
        energy, time_remain = 0, ddl

        pass

if __name__ == '__main__':
    s = Simulation('CIFAR10')
    print(s.profile_res)





