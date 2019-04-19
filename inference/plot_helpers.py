import numpy as np
import matplotlib.pyplot as plt
from inference.GP_helpers import generate_grid


def plotGP(model, x_train, title, res=30):

    X_guess = generate_grid(-2.0, 2.0, res)
    Y_pred = model.predict(X_guess)


    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, res), np.linspace(-2, 2, res), Y_pred[0].reshape(res, res))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(x_train[:, 0], x_train[:, 1])
    plt.title('Mean, ' + title)
    
    if 'signal field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")
    elif 'hardness field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")

    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, res), np.linspace(-2, 2, res), \
        np.sqrt(Y_pred[1].reshape(res, res)))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(x_train[:, 0], x_train[:, 1])
    plt.title('Stdev, ' + title)

def plotMOGP(model, x_train, output, title, res=30):

    X_guess = np.concatenate([generate_grid(-2.0, 2.0, res), np.ones((res*res, 1))*output], axis=1)
    Y_pred = model.predict(X_guess)


    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, res), np.linspace(-2, 2, res), Y_pred[0].reshape(res, res))
    plt.clabel(CS, inline=1, fontsize=10)
    #print(x_train[output].T[0])
    plt.plot(x_train[output].T[0], x_train[output].T[1])
    plt.title('Mean, ' + title)
    
    if 'signal field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")
    elif 'hardness field' in title:
        plt.scatter(points.T[0], points.T[1], c="r")

    plt.figure()
    CS = plt.contour(np.linspace(-2, 2, res), np.linspace(-2, 2, res), \
        np.sqrt(Y_pred[1].reshape(res, res)))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(x_train[output].T[0], x_train[output].T[1])
    plt.title('Stdev, ' + title)