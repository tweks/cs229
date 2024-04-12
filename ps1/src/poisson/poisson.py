import numpy as np
import util
import matplotlib.pyplot as plt


def plot(true_val, pred_val, save_path):
    plt.figure()
    plt.plot(true_val, pred_val, 'bo')
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    lim = np.concatenate((true_val, pred_val)).max() + .1
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.savefig(save_path)


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    _, num_features = x_train.shape

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr, theta_0=np.random.rand(num_features))
    clf.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    h_valid = clf.predict(x_valid)
    np.savetxt(save_path, h_valid)
    plot(y_valid, h_valid, f'{save_path}.png')
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        for num_iter in range(self.max_iter):
            h = self.predict(x)
            grad = (y - h) @ x
            if self.verbose:
                loss = np.linalg.norm(y - h)
                print(f'Epoch {num_iter}: loss {loss:.5f}')
            new_theta = self.theta + self.step_size * grad
            if np.linalg.norm(new_theta - self.theta) < self.eps:
                return
            self.theta = new_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
