from __future__ import division

from copy import deepcopy
from mcts import mcts
from functools import reduce
import operator
import numpy as np


class MCTSPlanner():
    def __init__(self):
        self.res = 30

    def policy(self, alpha, inference_model, loc):
        
        initialState = InformationState(loc[0], loc[1], inference_model.copy(), 1, inference_model.compute_entropy())
        mcts1 = mcts(iterationLimit=10000, explorationConstant=300)
        action = mcts1.search(initialState=initialState)
        nextSample = [action.x, action.y]

        return nextSample

class InformationState():
    def __init__(self, x_pos, y_pos, belief, horizon, initial_entropy):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.belief = belief
        self.observed_feature = 2
        self.horizon = horizon
        self.initial_entropy = initial_entropy
        #self.random_seed = np.random.rand()
        self.prev_d = [0, 0]

    def getPossibleActions(self):
        possibleActions = []
        displacements = np.array([[-0.2, 0], [0.2, 0], [0, -0.2], [0, 0.2]])
        #print(displacements)
        for d in displacements:
            if -2.0 <= self.x_pos + d[0] <= 2.0 and -2.0 <= self.y_pos + d[1] <= 2.0:
                #print(self.prev_d)
                #print(-1 * d)
                #if not np.array_equal(self.prev_d, -1 * d):
                possibleActions.append(Action(x=self.x_pos+d[0], y=self.y_pos+d[1], dx = d[0], dy = d[1]))#, random_seed=self.random_seed))
                #else:
                    #print('reverse!')
        return possibleActions

    def takeAction(self, action):
        print('Simulated action', action)
        newState = InformationState(self.x_pos, self.y_pos, self.belief.copy(), self.horizon, self.initial_entropy)

        observation = newState.belief.observe(np.array([[action.x, action.y, self.observed_feature]]))
        newState.belief.update(np.array([action.x, action.y, self.observed_feature]), observation, self.observed_feature)

        newState.x_pos = action.x
        newState.y_pos = action.y
        newState.prev_d = [action.x - self.x_pos, action.y - self.y_pos]
        newState.horizon = self.horizon - 1
        #newState.random_seed = self.random_seed

        return newState

    def isTerminal(self):
        return self.horizon <= 0

    def getReward(self):
        final_entropy = self.belief.compute_entropy()
        #print('Terminated at', [self.x_pos, self.y_pos], 'with entropy', final_entropy)
        return -1 * final_entropy


class Action():
    def __init__(self, x, y, dx, dy, random_seed=1.0):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.random_seed = random_seed

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.dx == other.dx and self.dy == other.dy and self.random_seed == other.random_seed

    def __hash__(self):
        return hash((self.x, self.y, self.dx, self.dy, self.random_seed))
