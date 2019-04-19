import GPy
import numpy as np
import scipy

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

def generate_rbfkern(dim, variance, lengthscale):
    return GPy.kern.RBF(dim, variance=variance, lengthscale=lengthscale)

def generate_grid(lb, ub, res):
    x1_grid = np.linspace(lb, ub, res)
    x2_grid = np.linspace(lb, ub, res)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.array([X1_grid, X2_grid]).reshape(2, -1).T
    return X_grid

