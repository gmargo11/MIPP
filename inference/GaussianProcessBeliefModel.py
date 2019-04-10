from inference.GP_helpers import GP, MOGP, generate_rbfkern, generate_grid
from inference.plot_helpers import plotGP, plotMOGP
import numpy as np
import math

from inference import InferenceModel


class GaussianProcessBeliefModel(): # InferenceModel
    def __init__(self):
        self.res = 20
        self.kernel = generate_rbfkern(2, 1.0, 0.3)
        self.entropy = 0

    def copy(self):
        newMe = GaussianProcessBeliefModel()
        newMe.x_train, newMe.y_train, newMe.num_features, newMe.entropy = self.x_train.copy(), self.y_train.copy(), self.num_features, self.entropy

        return newMe

    def load_environment(self, env, start_loc=[0, 0]):
        self.x_train, self.y_train, self.num_features = env.load_prior_data(start_loc)
        #self.entropy = -1 * len(self.x_train)
        #print(self.entropy)

    def update(self, x, y, feature):
        #print(np.min(abs(np.sum(self.x_train[feature] - [x[0:2]], axis=1))))
        #print('a')
        #print([x[0:2]])
        #print(self.x_train[feature])
        #print(self.x_train[feature] - [x[0:2]])
        #print(np.sum(abs(self.x_train[feature] - [x[0:2]]), axis=1))
        self.x_train[feature] = np.append(self.x_train[feature], [x[0:2]], axis=0)
        self.y_train[feature] = np.append(self.y_train[feature], y)

    def update_static(self, x, y, feature):
        self.x_train[feature] = np.append(self.x_train[feature], [x[0:2]], axis=0)
        self.y_train[feature] = np.append(self.y_train[feature], y)

    def observe(self, x):
        feature = int(x[0][2])
        #if x[0][0:2] in self.x_train[feature]:
        #    print(np.where(self.x_train[feature] == x[0][0:2]))
        #    return self.y_train[np.where(self.x_train[feature] == x[0][0:2])]
        #else:
        return 0
        
    def infer_independent_distribution(self, feature, res):
        independent_distribution = GP(self.x_train[feature], self.y_train[feature], self.kernel)
        return independent_distribution

    def display(self, feature, title):
        independent_distribution = self.infer_independent_distribution(feature=feature, res=20)
        plotGP(independent_distribution, self.x_train[feature], title=title, res=20)
    
    def generate_weights(self, S):
        W = np.reciprocal(np.sqrt(S[2, :]))
        return W

    def compute_entropy(self, res=21):
        #print(self.x_train[2], self.y_train[2])
        self.model = GP(self.x_train[2], self.y_train[2], self.kernel)

        x_candidates = generate_grid(-2.0, 2.0, res)
        mean, var = self.model.predict(x_candidates)
        self.entropy = np.sum(np.log(var))
        #self.entropy = mean + 0.3 * var

        #print(self.entropy)
        return self.entropy
    
    def evaluate_MSE(self, true_func, res=30):
        data = generate_grid(-2.0, 2.0, res)
        self.model = GP(self.x_train[2], self.y_train[2], self.kernel)
        m_pred, s_pred = self.model.predict(data)
        m_true = np.apply_along_axis(true_func, axis=1, arr=data).reshape(-1, 1)

        return np.sum((m_pred - m_true)**2) / (res * res)

