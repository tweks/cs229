import numpy as np
import util

DEBUG = False

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    print(f'******* {train_path}')
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    util.plot(x_train, y_train, None, f'{train_path}.png')

    # Train a logistic regression classifier
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    util.plot(x_valid, y_valid, lr.theta, f'{valid_path}.png')

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, lr.predict(x_valid))


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        n, dim = x.shape
        if self.theta is None:
            self.theta = np.zeros(dim)
        
        print_matrix(x, 'x')
        print_matrix(y, 'y')
        
        for i in range(self.max_iter):
            pred = self.predict(x)
            print_matrix(pred, 'pred')
            gradient = -1/n * x.T @ (y - pred)
            print_matrix(gradient, 'gradient')
            hessian = 1/n * x.T @ np.diag(pred * (1 - pred)) @ x
            print_matrix(hessian, 'hessian')
            delta = np.linalg.inv(hessian) @ gradient
            print_matrix(delta, 'delta')
            norm = np.linalg.norm(delta, 1)
            if self.verbose:
                pred_smoothed = pred - (pred == 1) * 1e-10
                loss = -1/n * (y @ np.log(pred_smoothed) + (1 - y) @ np.log(1 - pred_smoothed))
                print(f'epoch {i}: loss {loss:.5f} delta {norm:.5f}')
            self.theta = self.theta - delta
            if norm < self.eps:
                return

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        def sigmoid(x):
            return 1. / (1. + np.exp(-x))

        print_matrix(self.theta, 'theta')
        print_matrix(x, 'x')
        preds = sigmoid(x @ self.theta)
        print_matrix(preds, 'preds')
        return preds


def print_matrix(m, name):
    if DEBUG:
        print(f'{name}: {m.shape}')
        print(m)


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
