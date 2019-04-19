from data.SideInformationEnvironmentRandomGP import SideInformationEnvironmentRandomGP
from inference.GP_helpers import generate_grid
import numpy as np
import matplotlib.pyplot as plt

def plot_environment():
    env = new SideInformationEnvironmentRandomGP()
    num_features = len(env.func)
    res=20

    x_test = generate_grid(lb=-2.0, ub=2.0, res=res)

    y_test = [[] for i in range(num_features)]

    for i in range(num_features):
    	y_test[i] = [env.observe(xo, i) for xo in x_test]

    for i in range(num_features):
    	plt.figure()
	    CS = plt.contour(np.linspace(-2, 2, res), np.linspace(-2, 2, res), y_test[i].reshape(res, res))
	    plt.clabel(CS, inline=1, fontsize=10)
	    plt.plot(x_train[:, 0], x_train[:, 1])
	    plt.title('i')