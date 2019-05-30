from inference.GP_helpers import GP, MOGP, generate_rbfkern, generate_grid
from inference.plot_helpers import plotGP, plotMOGP
import numpy as np
import math

from inference import InferenceModel


class UncertainGPModel(): # InferenceModel
    def __init__(self):
        self.kernel = generate_rbfkern(2, 1.0, 0.3)

    def copy(self):
        newMe = UncertainGPModel()
        newMe.kernel = self.kernel
        newMe.x_train, newMe.y_train, newMe.num_features = self.x_train.copy(), self.y_train.copy(), self.num_features

        return newMe

    def load_environment(self, env, start_loc=[0, 0]):
        self.x_train, self.y_train, self.num_features = env.load_prior_data(start_loc)

    def update(self, x, y, feature):
        print(feature)
        self.x_train[feature] = np.append(self.x_train[feature], [x[0:2]], axis=0)
        self.y_train[feature] = np.append(self.y_train[feature], y)

    def observe(self, x):
        model = self.infer_joint_distribution(res=20)
        obs = model.predict(x)

        return obs[0] + np.random.randn() * np.sqrt(obs[1])
        

    def compute_belief_variance(self):
        priors = [[] for i in range(self.num_features)]
        for i in range(self.num_features):
            priors[i] = GP(self.x_train[i], self.y_train[i], self.kernel)
            #plotGP(priors[i], x_train[i], 'Feature '+str(i))

        m_obs = [[] for i in range(self.num_features)]
        m_exp = [[] for i in range(self.num_features)]
        s_obs = [[] for i in range(self.num_features)]


        ### sample variable distributions
        #for x in range(len(x_candidates)):
        for i in range(self.num_features):
            pred_obs = priors[2].predict(self.x_train[i])
            m_obs[i], s_obs[i] = pred_obs[0].flatten(), pred_obs[1].flatten()
            m_exp[i] = self.y_train[i]
            #print(pred[0].flatten())

        
        self.count = np.ones(self.num_features)



        for i in range(self.num_features):
            for j in range(len(self.x_train[i])):
                if i != 2 and s_obs[i][j] < 0.8: # if sufficiently confident about feature of interest where the side feature has been observed
                    self.count[i] += 1 # we know more about the relationship between those features


        self.feature_var = np.array([1.0 / math.sqrt(n) for n in self.count])

    def infer_joint_distribution(self, res):
        ### train GP priors
        
        x_candidates = generate_grid(-2, 2, res)


        priors = [[] for i in range(self.num_features)]
        for i in range(self.num_features):
            priors[i] = GP(self.x_train[i], self.y_train[i], self.kernel)
            #plotGP(priors[i], x_train[i], 'Feature '+str(i))

        #print(m[0, :])


        #for x in range(len(x_candidates)):
        #    for i in range(self.num_features):
        #        pred = priors[i].predict(np.array([x_candidates[x]]))
        #        m[i, x], s[i, x] = pred[0][0][0], pred[1][0][0]


        m = np.ndarray((self.num_features, res*res))
        s = np.ndarray((self.num_features, res*res))

        for i in range(self.num_features):
            pred = priors[i].predict(x_candidates)
            m[i], s[i] = pred[0].flatten(), pred[1].flatten()

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

    def compute_entropy(self, res=21):
        #model = self.infer_joint_distribution(res=res)
        #x_candidates = np.concatenate([generate_grid(-2.0, 2.0, res), np.ones((res*res, 1))*2], axis=1)
        self.compute_belief_variance()

        model = self.infer_independent_distribution(2, res=res)
        x_candidates = generate_grid(-2.0, 2.0, res)
        mean, var = model.predict(x_candidates)
        entropy = np.sum(np.log(var))

        
        beta = 100
        entropy += beta * np.sum(np.log(self.feature_var))
        print(self.count)
        print('Map entropy', np.sum(np.log(var)))
        print('Knowledge model entropy', beta * np.sum(np.log(self.feature_var)))

        print(entropy)
        return entropy

    def compute_variance(self, x, res=21):
        model = self.infer_joint_distribution(res=res)
        return model.predict(x)[1][0][0]

    def evaluate_MSE(self, true_func, res=21):
        data = np.concatenate([generate_grid(-2.0, 2.0, res), np.ones((res*res, 1))*2], axis=1)
        model = self.infer_joint_distribution(res=res)
        m_pred, s_pred = model.predict(data)
        m_true = np.apply_along_axis(true_func, axis=1, arr=data).reshape(-1, 1)

        return np.sum((m_pred - m_true)**2) / (res * res)
