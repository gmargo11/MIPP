from __future__ import division

from copy import deepcopy
from mcts import mcts
from functools import reduce
import operator
import numpy as np
import math
from scipy.stats import norm


class RandomPlanner():
    def __init__(self):
        self.res = 30
        self.observed_feature = 2

    def policy(self, alpha, inference_model, loc):
        
        stepsize = 0.1
        displacements = [[stepsize, 0], [-stepsize, 0], [0, stepsize], [0, -stepsize]] #[[stepsize * math.cos(t), stepsize * math.sin(t)] for t in np.linspace(0, 2*math.pi, 20)]
        candidates = []
        for d in displacements:
            if -2.0 <= loc[0] + d[0] <= 2.0 and -2.0 <= loc[1] + d[1] <= 2.0:
                candidates.append([loc[0] + d[0], loc[1] + d[1]])
        candidates = np.array(candidates)
        
        irand = np.random.randint(len(candidates))
        nextSample = candidates[irand]

        return nextSample
