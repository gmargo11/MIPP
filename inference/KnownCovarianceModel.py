from inference.GP_helpers import GP, MOGP, generate_rbfkern, generate_grid
from inference.plot_helpers import plotGP, plotMOGP
import numpy as np
import math

from inference import InferenceModel


class KnownCovarianceModel(): # InferenceModel
    def __init__(self):
        self.kernel = generate_rbfkern(2, 1.0, 0.3)
        self.res=41

    def copy(self):
        newMe = KnownCovarianceModel()
        newMe.kernel = self.kernel
        newMe.x_train, newMe.y_train, newMe.num_features = self.x_train.copy(), self.y_train.copy(), self.num_features
        newMe.cov = self.cov
        newMe.env = self.env

        return newMe

    def load_environment(self, env, start_loc=[0, 0]):
        self.x_train, self.y_train, self.num_features = env.load_prior_data(start_loc)

        # compute true covariance
        xs = generate_grid(-2.0, 2.0, res=self.res)
        y_true = np.zeros((self.num_features, self.res*self.res))
        
        for feature in range(self.num_features):
            y_true[feature, :] = np.apply_along_axis(env.func[feature], axis=1, arr=xs).reshape(-1, 1).flatten()

        Y_true = np.array([y_true[i] - np.mean(y_true[i]) for i in range(self.num_features)])
        print("loaded")
        self.cov = np.cov(Y_true)
        self.env = env


    def update(self, x, y, feature):
        print(feature)
        #self.x_train[feature] = np.append(self.x_train[feature], [x[0:2]], axis=0)
        #self.y_train[feature] = np.append(self.y_train[feature], y)

    def observe(self, x):
        return self.env.observe(x[0][0:2], 2)
        
    def infer_joint_distribution(self, res):
        ### train GP priors
        priors = [[] for i in range(self.num_features)]
        for i in range(self.num_features):
            priors[i] = GP(self.x_train[i], self.y_train[i], self.kernel)
            #plotGP(priors[i], x_train[i], 'Feature '+str(i))


        ### construct correlated joint distribution GP
        joint_distribution = MOGP(self.x_train, self.y_train, self.cov, self.kernel)
        #plotMOGP(joint_distribution, x_train, 2, 'Output 2')

        return joint_distribution

    def infer_independent_distribution(self, feature, res):
        independent_distribution = GP(self.x_train[feature], self.y_train[feature], self.kernel)
        return independent_distribution

    def display(self, feature, title):
        joint_distribution = self.infer_joint_distribution(res=21)
        plotMOGP(joint_distribution, self.x_train, output=feature, title=title, res=21)
    

    def compute_entropy(self, res=21):
        #model = self.infer_joint_distribution(res=res)

        #x_candidates = np.concatenate([generate_grid(-2.0, 2.0, res), np.ones((res*res, 1))*2], axis=1)
        #y_pred = model.predict(x_candidates)

        #entropy = np.sum(0.5 * np.log(math.sqrt(2 * math.pi * math.e) * y_pred[1][:][0]))
        #print(entropy)
        #return entropy
        return 0.0

    def evaluate_MSE(self, true_func, res=30):
        data = np.concatenate([generate_grid(-2.0, 2.0, res), np.ones((res*res, 1))*2], axis=1)
        model = self.infer_joint_distribution(res=res)
        m_pred, s_pred = model.predict(data)
        m_true = np.apply_along_axis(true_func, axis=1, arr=data).reshape(-1, 1)

        return np.sum((m_pred - m_true)**2) / (res * res)
