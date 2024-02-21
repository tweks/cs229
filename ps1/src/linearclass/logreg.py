import numpy as np
import util


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
    # util.plot(x_valid, y_valid, None, f'{valid_path}.png')

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    lr = LogisticRegression()
    # lr.fit(x_train, y_train)
    x_train_small = x_train[:2, :]
    y_train_small = y_train[:2]
    y_train_small[1] = 1.
    lr.fit(x_train_small, y_train_small)

    # Plot decision boundary on top of validation set set
    # util.plot(x_valid, y_valid, lr.theta, f'{valid_path}.png')

    # Use np.savetxt to save predictions on eval set to save_path
    # np.savetxt(save_path, lr.predict(x_valid))
    # *** END CODE HERE ***


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
        # *** START CODE HERE ***
        n, dim = x.shape
        if self.theta is None:
            self.theta = np.zeros((dim,))
        
        print_matrix(x, 'x')
        print_matrix(y, 'y')
        

        # while True:
        for i in range(10000):
            h = self.predict(x)
            print_matrix(h, 'h')
            dl = -1./n * x.T @ (y - h)
            print_matrix(dl, 'dl')
            H = 1./n * x.T @ np.diag(h * (1 - h)) @ x
            print_matrix(H, 'H')
            delta = np.linalg.inv(H) @ dl
            print_matrix(delta, 'delta')
            self.theta = self.theta - delta
            norm = np.linalg.norm(delta, 1)
            print(f'epoch {i}: {norm}')
            if norm < self.eps:
                return
        # *** END CODE HERE ***

    def deprecated_fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, dim = x.shape
        if self.theta is None:
            self.theta = np.zeros((dim, 1))
        
        y = y[:, np.newaxis]
        
        print_matrix(x, 'x')
        print_matrix(y, 'y')
        
        x_outer = x[:, :, np.newaxis] * x[:, np.newaxis, :]
        print_matrix(x_outer, 'x_outer')

        # while True:
        for i in range(10000):
            h = self.predict(x)
            print_matrix(h, 'h')
            dl = -1./n * np.sum(x * (y - h), axis=0)
            dl = dl[:, np.newaxis]
            print_matrix(dl, 'dl')
            pr = np.expand_dims(h * (1 - h), axis=-1)
            print_matrix(pr, 'pr')
            H = 1./n * np.sum(pr * x_outer, axis=0)
            print_matrix(H, 'H')
            delta = np.linalg.inv(H) @ dl
            print_matrix(delta, 'delta')
            self.theta = self.theta - delta
            norm = np.linalg.norm(delta, 1)
            print(f'epoch {i}: {norm}')
            if norm < self.eps:
                return
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        def sigmoid(x):
            return 1. / (1. + np.exp(-x))

        print_matrix(self.theta, 'theta')
        print_matrix(x, 'x')
        preds = sigmoid(x @ self.theta.T)
        print_matrix(preds, 'preds')

        return preds
        # *** END CODE HERE ***

def print_matrix(m, name):
    print(f'{name}: {m.shape}')
    print(m)
    # pass


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
