import numpy as np
import util
import sys

from linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    _, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    xlim, ylim = util.get_lim([x_train, x_test], 4)

    util.plot(x_train, t_train, None, f'{train_path}.png', xlim, ylim)

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    lr = LogisticRegression(verbose=False)
    lr.fit(x_train, t_train)
    util.plot(x_test, t_test, lr.theta, f'{output_path_true}.png', xlim, ylim)
    np.savetxt(output_path_true, lr.predict(x_test))

    # Part (b): Train on y-labels and test on true labels
    lr = LogisticRegression(verbose=False)
    lr.fit(x_train, y_train)
    util.plot(x_test, t_test, lr.theta, f'{output_path_naive}.png', xlim, ylim)
    np.savetxt(output_path_naive, lr.predict(x_test))

    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
