import numpy as np
import time
import math
from MIPPutil import GP, MOGP, plotGP, plotMOGP, import_data, generate_grid, generate_weights, generate_rbfkern, compute_novelties, compute_utilities, signal_field

res = 30

def infer_joint_distribution(x_train, y_train, num_features):

	### train GP priors
	priors = [[] for i in range(num_features)]
	for i in range(num_features):
	    priors[i] = GP(x_train[i], y_train[i], generate_rbfkern(2, 1.0, 0.3))
	    #plotGP(priors[i], x_train[i], 'Feature '+str(i))

	m = np.ndarray((num_features, res*res))
	s = np.ndarray((num_features, res*res))

	x_candidates = generate_grid(res)

	### sample variable distributions
	for x in range(len(x_candidates)):
	    for i in range(num_features):
	        pred = priors[i].predict(np.array([x_candidates[x]]))
	        m[i, x], s[i, x] = pred[0][0][0], pred[1][0][0]

	### compute weighted covariance
	W = generate_weights(s)
	X = np.array([m[i] - np.mean(m[i]) for i in range(num_features)])

	w_cov = np.cov(X, aweights=W)

	### use lasso to estimate a sparse precision matrix? (GGM model estimation)

	### construct correlated joint distribution GP
	joint_distribution = MOGP(x_train, y_train, w_cov, generate_rbfkern(2, 1.0, 0.3))
	#plotMOGP(joint_distribution, x_train, 2, 'Output 2')

	return joint_distribution

def explore(x_train, y_train, num_features):
    
    joint_distribution = infer_joint_distribution(x_train, y_train, num_features)
    independent_distribution = GP(x_train[2], y_train[2], generate_rbfkern(2, 1.0, 0.3))

	### compute predicted novelty metric
    x_candidates = np.concatenate([generate_grid(res), np.ones((res*res, 1))*0], axis=1)
    novelties = compute_novelties(joint_distribution, x_train)
    nextSample = x_train[1][np.argmax(novelties)]
    #nextSample = x_candidates[np.random.randint(len(x_candidates))]

    ### Sample and update model
    return nextSample, joint_distribution, independent_distribution

def exploit(x_train, y_train, num_features):

    joint_distribution = infer_joint_distribution(x_train, y_train, num_features)
    independent_distribution = GP(x_train[2], y_train[2], generate_rbfkern(2, 1.0, 0.3))

    x_candidates = np.concatenate([generate_grid(res), np.ones((res*res, 1))*2], axis=1)
    utilities = compute_utilities(joint_distribution, independent_distribution, x_candidates)
    nextSample = x_candidates[np.argmax(utilities)]

    return nextSample, joint_distribution, independent_distribution


def follow_complete_policy():
    alpha = 1.0
    x_train, y_train, num_features = import_data()
    observation_function = signal_field

    for i in range(11):
        if i > 6: #np.random.uniform() > alpha:
            print('exploit')
            nextSample, joint_distribution, independent_distribution = exploit(x_train, y_train, num_features)
            observation = observation_function(nextSample)
            x_train[2] = np.append(x_train[2], [nextSample[0:2]], axis=0)
            y_train[2] = np.append(y_train[2], observation)
        else:
            print('explore')
            nextSample, joint_distribution, independent_distribution = explore(x_train, y_train, num_features)
            observation = observation_function(nextSample)
            x_train[2] = np.append(x_train[2], [nextSample[0:2]], axis=0)
            y_train[2] = np.append(y_train[2], observation)
        print(nextSample)
        if i % 5 == 0:
        	plotMOGP(joint_distribution, x_train, 2, 'Output 2')
        	plotMOGP(independent_distribution, x_train, 2, 'Output 2')
        alpha = alpha * 1.00
    



if __name__ == '__main__':
    follow_complete_policy()
    time.sleep(100)