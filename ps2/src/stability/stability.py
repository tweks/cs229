# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = - (Y - probs).dot(X)

    return grad


def calc_loss(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    loss = - (Y * np.log(probs) + (1 - Y) * np.log(1 - probs))
    return loss.sum() / count


def predict(theta, x):
    x.insert(0, 1)
    x = np.array(x)
    return x.dot(theta)


def logistic_regression(X, Y, save_path, lr=0.1):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = lr

    i = 0
    loss_fig, loss_ax = plt.subplots()
    points_fig, points_ax = plt.subplots()

    maxiter = 100000
    checkpoint = 100
    plot_x, plot_y = [], []
    for i in range(1, maxiter+1):
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % checkpoint == 0:
            loss = calc_loss(X, Y, theta)
            print(f'Finished {i} iterations. Delta: {np.linalg.norm(prev_theta - theta)}. Loss: {loss}. Theta: {theta}. Grad: {grad}')
            plot_x.append(i)
            plot_y.append(loss)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    util.plot_boundary(points_ax, X, theta, 'red')
    util.plot_points(points_ax, X[:, 1:], Y)
    points_fig.savefig(save_path)
    loss_ax.plot(plot_x, plot_y, c='red', linewidth=1)
    loss_ax.set_xlabel('iteration')
    loss_ax.set_ylabel('loss')
    loss_fig.savefig(f'loss_{save_path}')


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, 'ds1_a.png', lr=0.1)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, 'ds1_b.png', lr=0.1)


if __name__ == '__main__':
    main()
