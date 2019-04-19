from __future__ import division

from copy import deepcopy
from mcts import mcts
from functools import reduce
import operator
import numpy as np
import math
from scipy.stats import norm

from inference.GP_helpers import generate_grid


class BanditPlanner():
    def __init__(self):
        self.res = 30
        self.observed_feature = 2

    def policy(self, alpha, inference_model, loc):
        
        stepsize = 0.5
        candidates = generate_grid(-2.0, 2.0, self.res)

        ### Entropy minimization acquisition function
        '''
        min_entropy = inference_model.compute_entropy()
        print('initial entropy')
        print(inference_model.compute_entropy())
        for candidate in candidates:
            print(candidate)
            predictedBelief = inference_model.copy()
            observation = predictedBelief.observe(np.array([[candidate[0], candidate[1], self.observed_feature]]))
            predictedBelief.update(np.array([candidate[0], candidate[1], self.observed_feature]), observation, self.observed_feature)
            print(predictedBelief.compute_entropy())
            if predictedBelief.compute_entropy() <= min_entropy:
                min_entropy = predictedBelief.compute_entropy()
                nextSample = candidate
        '''

        # UCB acquisition function
        mean, var = inference_model.infer_independent_distribution(feature=2, res=30).predict(candidates)
        ucb = mean + 2.0 * var
        #ucb = (mean - 0.0) * norm.cdf((mean - 0.0) / np.sqrt(var)) + np.sqrt(var) * norm.pdf((mean - 0.0) / np.sqrt(var))
        nextSample = candidates[np.argmax(ucb)]

        return nextSample
