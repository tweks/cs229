import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    x_valid_int, _ = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)

    # Plot decision boundary on validation set
    util.plot(x_valid_int, y_valid, gda.theta, f'gda_{valid_path}.png')

    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, gda.predict(x_valid))
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        n_examples, _ = x.shape
        util.print_matrix(x, 'x')
        util.print_matrix(y, 'y')
        phi = y @ y / n_examples
        util.print_matrix(phi, 'phi')
        mu_0 = ((1 - y) @ x) / ((1 - y) @ (1 - y))
        util.print_matrix(mu_0, 'mu_0')
        mu_1 = (y @ x) / (y @ y)
        util.print_matrix(mu_1, 'mu_1')
        mu = np.array([mu_0, mu_1])
        util.print_matrix(mu, 'mu')
        mu_y = mu[y.astype(int)]
        util.print_matrix(mu_y, 'mu_y')
        sigma = (x - mu_y).T @ (x - mu_y) / n_examples
        util.print_matrix(sigma, 'sigma')

        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        theta = sigma_inv @ (mu_1 - mu_0)
        util.print_matrix(theta, 'theta')
        theta_0 = -1/2 * mu_1.T @ sigma_inv @ mu_1 + 1/2 * mu_0.T @ sigma_inv @ mu_0 + np.log(phi / (1-phi))
        util.print_matrix(theta_0, 'theta_0')
        self.theta = np.insert(theta, 0, theta_0)
        util.print_matrix(self.theta, 'self.theta')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return util.sigmoid(x @ self.theta[1:] + self.theta[0])
        # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
