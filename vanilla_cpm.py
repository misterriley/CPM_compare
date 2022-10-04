import numpy
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from main import CPMMasker
import data_loader
from sort_and_regress import p

import numpy as np
from main import convert_to_wide

CPM_THRESHOLD = 0.05
N_FOLDS = 10
N_REPEATS = 20
VERBOSITY = 10


def main():
    ds = data_loader.get_sadie_marie_2_data_sets(as_r=False)
    out_df = pd.DataFrame(columns=["descriptor", "mask_type", "pseudo_r2"])
    for d in ds:
        p(-1, "Starting CPM for {}...".format(d.descriptor), 2, verbosity=VERBOSITY)
        x = d.x
        y = d.y

        if np.unique(y).size == 2:
            y = np.array([1 if i == np.unique(y)[1] else 0 for i in y])

        x = convert_to_wide(x)
        for mask_type in ["all", "positive", "negative"]:
            p(-1, "Calculating spearman rho for {}...".format(mask_type), 3, verbosity=VERBOSITY)
            cpm_spearman = get_cpm_spearman_average(x, y, mask_type, n_repeats=N_REPEATS)
            out_df = out_df.append({"descriptor": d.get_descriptor(), "mask_type": mask_type,
                                    "spearman_rho": cpm_spearman},
                                   ignore_index=True)
    p(-1, "Saving results...", 1, verbosity=VERBOSITY)
    out_df.to_csv("cpm_rho_values.csv")


def get_cpm_spearman_average(x, y, mask_type, n_repeats=1):
    rhos = np.ndarray((n_repeats,))
    for i in range(n_repeats):
        p(-1, "Calculating spearman rho repeat {}/{}...".format(i+1, n_repeats), 4, verbosity=VERBOSITY)
        kf = KFold(n_splits=N_FOLDS, shuffle=True)
        y_hat = np.ndarray(y.shape[0])
        binary = False
        for split_index, (train_index, test_index) in enumerate(kf.split(x)):
            p(-1, "Fitting fold {}/{}...".format(split_index+1, N_FOLDS), 5, verbosity=VERBOSITY)
            reg, masker = fit(x[train_index], y[train_index], mask_type)
            x_test, n_params = masker.get_x(CPM_THRESHOLD, mask_type, x[test_index])
            if isinstance(reg, LinearRegression):
                y_hat[test_index] = reg.predict(x_test)
            else:
                binary = True
                y_hat[test_index] = reg.predict_proba(x_test)[:, 1]
        if binary:
            d_ll = np.sum(np.log(y_hat[y == 1])) + np.sum(np.log(1 - y_hat[y == 0]))
            y_mean = np.mean(y)
            d_ll_null = y.shape[0] * (y_mean * np.log(y_mean) + (1 - y_mean) * np.log(1 - y_mean))
            rhos[i] = 1 - d_ll / d_ll_null
        else:
            rhos[i] = spearmanr(y, y_hat)[0]
        p(-1, "Spearman rho: {}".format(rhos[i]), 4, verbosity=VERBOSITY)
    ret = np.mean(rhos)
    p(-1, "Average spearman rho: {}".format(ret), 3, verbosity=VERBOSITY)
    return ret


def fit(x, y, mask_type):
    binary = False # numpy.unique(y).size == 2
    masker = CPMMasker(x, y, binary=binary)
    xs, n_vals = masker.get_x(CPM_THRESHOLD, mask_type)
    clf = LogisticRegression(penalty='none', class_weight='balanced') if binary else LinearRegression()
    clf.fit(xs, y)
    return clf, masker


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
