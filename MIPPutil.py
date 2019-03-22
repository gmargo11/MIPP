import GPy
from libpgm.pgmlearner import PGMLearner
import numpy as np
import scipy
from scipy.stats import wishart, norm
from sklearn.model_selection import GridSearchCV
from inverse_covariance import QuicGraphicalLasso
from inverse_covariance.plot_util import trace_plot
import matplotlib.pyplot as plt
import math
import json
import networkx as nx
import pygraphviz
import time


def generate_grid(size):
    x1_grid = np.linspace(-2, 2, size)
    x2_grid = np.linspace(-2, 2, size)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.array([X1_grid, X2_grid]).reshape(2, -1).T
    return X_grid

points = np.array([[-0.75, -0.75],  [-0.7, 0.0], [-0.7, 0.9], [1.5, -1.3]])

def signal_field(x, points=points):
    #points = [[-0.5, 0.5], [0.2, 0.2], [0.3, -0.6], [0.9, 0.3], [-0.1, -0.8]]
    strength = 0
    for point in points:
        dist = math.sqrt((x[0] - point[0])**2 + (x[1] - point[1])**2)
        strength += math.exp(-dist*3) * 2
    return strength

def depth_field(x):
    return -1 * x[0]

def hardness_field(x):
    return (signal_field(x, points=points) - x[0]) * 2 #+ np.random.rand()*0.1

def random_field(x):
    return 0.1 + np.random.randn() * 0.01

def import_data():
    ''' inputs: none
        outputs:
            x_task: a numpy array of input-values for each sample for each function
            y_task: a numpy array of output-values for each sample from each function
            num_funcs: the number of state variables in the dataset
    '''


    func = [hardness_field, depth_field, signal_field, random_field]
    names = ['hardness field', 'depth field', 'signal field', 'random field']
    n_sample = [225, 400, 1, 100]
    num_funcs = len(func)
    x_task = np.array([None for i in range(num_funcs)])
    y_task = np.array([None for i in range(num_funcs)])

    ### generate training data from functions
    for i in range(num_funcs):
        if i == 2: 
            x_task[i] = np.array([np.random.rand(n_sample[i]) * -1, np.random.rand(n_sample[i]) * -1]).T
        #elif i == 0:
        #    x_task[i] = np.array([np.random.rand(n_sample[i]) * -1, np.random.rand(n_sample[i]) * 2 -1]).T
        else:
            x_task[i] = generate_grid(int(math.sqrt(n_sample[i])))
        y_task[i] = np.array([func[i](xp) for xp in x_task[i]])
        
    return x_task, y_task, num_funcs

def GP(x, y, kern):
    ''' inputs:
            x: a numpy array of input-values for a single state variable
            y: a numpy array of output-values for a single state variable
            kern: a GPy kernel specifying an appropriate lengthscale and variance
        outputs:
            GP: a GPy GPRegression object
    '''
    return GPy.models.GPRegression(x, np.array(y).reshape(-1, 1), kern, noise_var=0.02)

def MOGP(x, y, cov, kern):
    ''' inputs:
            x: a numpy array of input-values for each state variable
            y: a numpy array of output-values for each state variable
            cov: a covariance matrix specifying the relationships between state variables
            kern: a GPy kernel specifying an appropriate lengthscale and variance
        outputs:
            MOGP: a GPy GPRegression object
    '''
    lcm1 = GPy.util.multioutput.ICM(input_dim=2, num_outputs=len(y), W_rank = len(y), kernel=kern)
    lcm1.B.W = scipy.linalg.sqrtm(np.mat(cov))
    #lcm1.B.W = scipy.linalg.sqrtm(np.mat(Bc))
    lcm1.B.kappa = np.zeros((1, len(y)))


    multiX = np.concatenate([np.concatenate([x[i], np.ones((len(x[i]), 1))*i], axis=1) for i in range(len(y))])
    multiY = np.concatenate([y[i] for i in range(len(y))]).reshape(-1, 1)

    m = GPy.models.GPRegression(multiX, multiY, lcm1, noise_var=0.02)

    return m

def plotGP(model, x_train, title):
    size=30

    X_guess = generate_grid(size)
    Y_pred = model.predict(X_guess)


    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, size), np.linspace(-2, 2, size), Y_pred[0].reshape(size, size))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(x_train[:, 0], x_train[:, 1])
    plt.title('Mean, ' + title)
    
    if 'signal field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")
    elif 'hardness field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")

    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, size), np.linspace(-2, 2, size), \
        np.sqrt(Y_pred[1].reshape(size, size)))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(x_train[:][0], x_train[:][1])
    plt.title('Stdev, ' + title)

def plotMOGP(model, x_train, output, title):
    size=50

    X_guess = np.concatenate([generate_grid(size), np.ones((size*size, 1))*output], axis=1)
    Y_pred = model.predict(X_guess)


    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, size), np.linspace(-2, 2, size), Y_pred[0].reshape(size, size))
    plt.clabel(CS, inline=1, fontsize=10)
    #print(x_train[output].T[0])
    plt.plot(x_train[output].T[0], x_train[output].T[1])
    plt.title('Mean, ' + title)
    
    if 'signal field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")
    elif 'hardness field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")

    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, size), np.linspace(-2, 2, size), \
        np.sqrt(Y_pred[1].reshape(size, size)))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(x_train[output].T[0], x_train[output].T[1])
    plt.title('Stdev, ' + title)


def generate_weights(S):
    W = np.reciprocal(np.sqrt(S[2, :]))
    return W

def compute_novelties(model, x_train, feature=1):

    novelties = np.array([0.0 for candidate in x_train[feature]])
    y_pred = model.predict(np.concatenate([x_train[feature], np.ones((len(x_train[feature]), 1))*feature], axis=1))[:][0][:]
    y_observed = model.predict(np.concatenate([x_train[2], np.ones((len(x_train[2]), 1))*feature], axis=1))[:][0][:]
    
    for j in range(len(y_pred)):
        novelties[j] = np.amin([abs(y_pred[j][0] - y_observed[i][0]) for i in range(len(y_observed))])

    print(scipy.stats.kstest(novelties, 'uniform')) # measure of how distributed our samples are

    plt.figure()
    size = int(math.sqrt(len(novelties)))
    plt.contour(np.linspace(-1, 1, size), np.linspace(-1, 1, size), \
        novelties.reshape(size, size))

    return novelties


'''

def compute_novelties(model, y_task, x_candidates):
    novelties = np.array([[0.0 for feature in range(len(y_task))] for candidate in x_candidates[0]])
    y_pred = [model.predict(x_candidates[feature])[:][0][:] for feature in range(len(y_task))]

    for j in range(len(y_pred[0])):
        for feature in range(len(y_task)): #range(len(y_pred)):
            novelties[j][feature] = np.amin([abs(y_pred[feature][j] - y_task[feature][i]) for i in range(len(y_task[feature]))])
    

    total_novelties = [max(nf) for nf in novelties]

    return total_novelties

'''


def compute_utilities(inference_model, observation_model, x_candidates):

    utilities = np.array([0.0 for candidate in x_candidates])
    y_pred = inference_model.predict(x_candidates)
    y_pred_obs = observation_model.predict(x_candidates[:, 0:2])
    
    for j in range(len(y_pred[0])):
        #utilities[j] = y_pred[0][j][0] * math.sqrt(y_pred_obs[1][j][0])
        utilities[j] = norm.cdf((y_pred[0][j][0] - 0.45) / y_pred[1][j][0]) * y_pred_obs[1][j][0]
    
    plt.figure()
    size = int(math.sqrt(len(utilities)))
    plt.contour(np.linspace(-1, 1, size), np.linspace(-1, 1, size), \
        utilities.reshape(size, size))

    return utilities

def generate_rbfkern(dim, variance, lengthscale):
    return GPy.kern.RBF(dim, variance=variance, lengthscale=lengthscale)