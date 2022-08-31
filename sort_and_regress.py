import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import data_loader
import numpy as np
from main import convert_to_wide, CPMMasker

N_REPEATS = 20
N_SPLITS = 10
MAX_ENTRIES = 200
VERBOSITY = 4


def p(s, threshold, verbosity=VERBOSITY):
    if verbosity >= threshold:
        print(s)


def regress_on_best_entries(x_train, y_train, x_test, y_test, mask_type, n_entries=None):
    masker = CPMMasker(x_train, y_train)
    coefficient_order = masker.get_coef_order(mask_type)

    ret = None
    if n_entries is None:
        ret = np.ndarray(shape=(MAX_ENTRIES + 1, y_test.shape[0]), dtype=object)
        ret[0, :] = y_train.mean()

    if n_entries == 0:
        return np.repeat(y_train.mean(), y_test.shape[0])

    dim_range = np.arange(1, MAX_ENTRIES + 1) if n_entries is None else [n_entries]

    for i in dim_range:
        p("Dimension {}/{}".format(i, MAX_ENTRIES), 5)
        dof = x_train.shape[0] - i - 1
        if dof < 0:
            break
        mask = np.zeros(x_train.shape[1], dtype=bool)
        mask[coefficient_order[:i]] = True

        reg = LinearRegression()
        reg.fit(x_train[:, mask], y_train)
        y_hat = reg.predict(x_test[:, mask])

        if n_entries is None:
            ret[i] = y_hat
        else:
            return y_hat

    return ret


def main():
    ds = data_loader.get_imagen_data_sets(file_c="mats_mid_bsl.mat",
                                          # y_col_c="csi_c_sum_fu2",
                                          as_r=False,
                                          clean_data=True)
    for d in ds:
        x = d.x
        y = d.y

        x = convert_to_wide(x)
        p("descriptor: {}".format(d.descriptor), 1)
        for mask_type in ["all", "positive", "negative"]:
            p("Mask type: {}".format(mask_type), 2)
            results = np.array([one_outer_repeat(x, y, i, mask_type) for i in range(N_REPEATS)])
            out_df = pd.DataFrame({"run_index": np.arange(1, N_REPEATS + 1),
                                   "spearman r": results[:, 0],
                                   "best_dim": results[:, 1],
                                   "mask_type": mask_type})
            out_df.loc[len(out_df)] = ["average",
                                       out_df["spearman r"].mean(),
                                       out_df["best_dim"].mean(),
                                       mask_type]

            out_df.to_csv("s_and_r_{}_{}.csv".format(d.get_descriptor(), mask_type), index=False)
            p("{} {} {}".format(d.get_descriptor(), np.mean(results[:, 0]), np.mean(results[:, 1])), 2)


def one_outer_repeat(x, y, repeat_index, mask_type):
    p("Repeat {}/{}".format(repeat_index + 1, N_REPEATS), 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    folds = list(kf.split(x))
    y_hat = np.zeros(y.shape[0])
    for i, (train_index, test_index) in enumerate(folds):
        p("Outer Fold {}/{}".format(i + 1, kf.n_splits), 4)
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        best_dim = get_best_dim(x_train, y_train, mask_type)
        y_hat[test_index] = regress_on_best_entries(x_train, y_train, x_test, y_test, mask_type, best_dim)
    best_dim = get_best_dim(x, y, mask_type, folds)
    p("Best dim: {}".format(best_dim), 3)
    return spearmanr(y_hat, y)[0], best_dim


def get_best_dim(x, y, mask_type, folds=None):
    if folds is None:
        folds = list(KFold(n_splits=N_SPLITS, shuffle=True).split(x))
    y_hat = np.zeros((MAX_ENTRIES + 1, y.shape[0]))
    for i, (train_index, test_index) in enumerate(folds):
        p("Fold {}/{}".format(i + 1, len(folds)), 2)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_hat[:, test_index] = regress_on_best_entries(x_train, y_train, x_test, y_test, mask_type)
    spearman_correlations = [spearmanr(y_hat[i], y)[0] for i in range(MAX_ENTRIES + 1)]
    best = np.nanargmax(spearman_correlations)
    return best


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    main()
