from inference.GP_helpers import GP, MOGP, generate_rbfkern, generate_grid
from inference.plot_helpers import plotGP, plotMOGP
import numpy as np
import math

from inference import InferenceModel


class DiscreteGaussianBeliefModel(): # InferenceModel
    def __init__(self):
        self.res = 20
        self.kernel = generate_rbfkern(2, 1.0, 0.3)
        self.entropy = 1.0

    def copy(self):
        newMe = DiscreteGaussianBeliefModel()
        newMe.x_train, newMe.y_train, newMe.num_features, newMe.entropy = self.x_train.copy(), self.y_train.copy(), self.num_features, self.entropy

        return newMe

    def load_environment(self, env):
        self.x_train, self.y_train, self.num_features = env.load_prior_data()
        #self.entropy = -1 * len(self.x_train)
        #print(self.entropy)

    def update(self, x, y, feature):
        #print(np.min(abs(np.sum(self.x_train[feature] - [x[0:2]], axis=1))))
        #print('a')
        #print([x[0:2]])
        #print(self.x_train[feature])
        #print(self.x_train[feature] - [x[0:2]])
        #print(np.sum(abs(self.x_train[feature] - [x[0:2]]), axis=1))
        if np.min(np.sum(abs(self.x_train[feature] - [x[0:2]]), axis=1)) > 0.01:
            self.entropy = self.entropy * 0.95
        else:
            print('entropy not decreased!')
        print(self.entropy)
        self.x_train[feature] = np.append(self.x_train[feature], [x[0:2]], axis=0)
        self.y_train[feature] = np.append(self.y_train[feature], y)

    def observe(self, x):
        feature = int(x[0][2])
        if x[0][0:2] in self.x_train[feature]:
            return self.x_train[feature]
        else:
            return np.random.randn()
        
    def infer_joint_distribution(self, res):
        ### train GP priors
        priors = [[] for i in range(self.num_features)]
        for i in range(self.num_features):
            priors[i] = GP(self.x_train[i], self.y_train[i], self.kernel)
            #plotGP(priors[i], x_train[i], 'Feature '+str(i))

        m = np.ndarray((self.num_features, res*res))
        s = np.ndarray((self.num_features, res*res))

        x_candidates = generate_grid(-2, 2, res)

        ### sample variable distributions
        for x in range(len(x_candidates)):
            for i in range(self.num_features):
                pred = priors[i].predict(np.array([x_candidates[x]]))
                m[i, x], s[i, x] = pred[0][0][0], pred[1][0][0]

        ### compute weighted covariance
        W = self.generate_weights(s)
        X = np.array([m[i] - np.mean(m[i]) for i in range(self.num_features)])

        w_cov = np.cov(X, aweights=W)

        ### use lasso to estimate a sparse precision matrix? (GGM model estimation)

        ### construct correlated joint distribution GP
        joint_distribution = MOGP(self.x_train, self.y_train, w_cov, self.kernel)
        #plotMOGP(joint_distribution, x_train, 2, 'Output 2')

        return joint_distribution

    def infer_independent_distribution(self, feature, res):
        independent_distribution = GP(self.x_train[feature], self.y_train[feature], self.kernel)
        return independent_distribution

    def display(self, feature, title):
        joint_distribution = self.infer_joint_distribution(res=20)
        plotMOGP(joint_distribution, self.x_train, output=feature, title=title, res=20)
    
    def generate_weights(self, S):
        W = np.reciprocal(np.sqrt(S[2, :]))
        return W

    def compute_entropy(self, res=20):
        #print(self.entropy)
        return self.entropy
