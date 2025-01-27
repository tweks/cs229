import matplotlib.pyplot as plt
import numpy as np


DEBUG_LEVEL = 1


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, xlim=None, ylim=None, correction=1.0, alpha=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2, alpha=alpha)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2, alpha=alpha)

    if theta is not None:
        # Plot decision boundary (found by solving for theta^T x = 0)
        if xlim is None:
            x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        else:
            x1 = np.arange(*xlim, 0.01)
        x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
            + np.log((2 - correction) / correction) / theta[2])
        plt.plot(x1, x2, c='red', linewidth=2)
    if xlim is None:
        xlim = (x[:, -2].min()-.1, x[:, -2].max()+.1)
    if ylim is None:
        ylim = (x[:, -1].min()-.1, x[:, -1].max()+.1)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


def get_lim(arrays, expand_by=.1):
    x = np.concatenate(arrays)
    xlim = (x[:, -2].min() - expand_by, x[:, -2].max() + expand_by)
    ylim = (x[:, -1].min() - expand_by, x[:, -1].max() + expand_by)
    return xlim, ylim


def print_matrix(m, name):
    if DEBUG_LEVEL >= 1: print(f'{name}: {m.shape} {m.dtype}')
    if DEBUG_LEVEL >= 2: print(m)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def loss(pred, y):
    n, = y.shape
    pred_smoothed = pred - (pred == 1) * 1e-10
    loss = -1/n * (y @ np.log(pred_smoothed) + (1 - y) @ np.log(1 - pred_smoothed))
    return loss


def accuracy(pred, y):
    n, = y.shape
    return np.sum(1 - np.logical_xor(pred > 0.5, y)) / n


def print_stats(pred, y, name):
    l = loss(pred, y)
    acc = accuracy(pred, y)
    print(f'{name}: accuracy: {acc:.5f} loss: {l:.5f}')