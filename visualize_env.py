from data.SideInformationEnvironmentRandomGP import SideInformationEnvironmentRandomGP
from inference.GP_helpers import generate_grid
import numpy as np
import matplotlib.pyplot as plt

def plot_environment():
    env = SideInformationEnvironmentRandomGP(points=[])
    num_features = len(env.func)
    res=20

    x_test = generate_grid(lb=-2.0, ub=2.0, res=res)

    y_test = [[] for i in range(num_features)]

    for i in range(num_features):
        y_test[i] = np.array([env.observe(xo, i) for xo in x_test])


    titles = ["Temperature", "Sea Floor Depth", "Luminosity", "Ocean Current Intensity"]
    plt.figure()
    for i in range(num_features):
        plt.subplot(1, num_features, i+1)
        CS = plt.contourf(np.linspace(0, 600, res), np.linspace(0, 600, res), y_test[i].reshape(res, res))
        #plt.clabel(CS, inline=1, fontsize=10)
        plt.title(titles[i])

    plt.show()


if __name__ == "__main__":
    plot_environment()