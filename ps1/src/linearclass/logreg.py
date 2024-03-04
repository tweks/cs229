import numpy as np
import util

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    xlim, ylim = util.get_lim(x_train, x_valid)

    util.plot(x_train, y_train, None, f'{train_path}.png', xlim, ylim)
    util.plot(x_train[y_train == 0], y_train[y_train == 0], None, f'{train_path}_0.png', xlim, ylim)
    util.plot(x_train[y_train == 1], y_train[y_train == 1], None, f'{train_path}_1.png', xlim, ylim)
    util.plot(x_valid[y_valid == 0], y_valid[y_valid == 0], None, f'{valid_path}_0.png', xlim, ylim)
    util.plot(x_valid[y_valid == 1], y_valid[y_valid == 1], None, f'{valid_path}_1.png', xlim, ylim)

    # Train a logistic regression classifier
    lr = LogisticRegression(verbose=False)
    lr.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    util.plot(x_valid, y_valid, lr.theta, f'logreg_{valid_path}.png', xlim, ylim)

    pred_train = lr.predict(x_train)
    loss_train = util.loss(pred_train, y_train)
    print(f'Loss for {train_path}: {loss_train:.5f}')

    pred_valid = lr.predict(x_valid)
    loss_valid = util.loss(pred_valid, y_valid)
    print(f'Loss for {valid_path}: {loss_valid:.5f}')

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
        
        util.print_matrix(x, 'x')
        util.print_matrix(y, 'y')
        
        for i in range(self.max_iter):
            pred = self.predict(x)
            util.print_matrix(pred, 'pred')
            gradient = -1/n * x.T @ (y - pred)
            util.print_matrix(gradient, 'gradient')
            hessian = 1/n * x.T @ np.diag(pred * (1 - pred)) @ x
            util.print_matrix(hessian, 'hessian')
            delta = np.linalg.inv(hessian) @ gradient
            util.print_matrix(delta, 'delta')
            norm = np.linalg.norm(delta, 1)
            if self.verbose:
                loss = util.loss(pred, y)
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

        util.print_matrix(self.theta, 'theta')
        util.print_matrix(x, 'x')
        preds = util.sigmoid(x @ self.theta)
        util.print_matrix(preds, 'preds')
        return preds


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
