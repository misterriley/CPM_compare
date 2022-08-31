import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from main import CPMMasker
import data_loader
from sort_and_regress import p

import numpy as np
from main import convert_to_wide

CPM_THRESHOLD = 0.05
N_FOLDS = 10
N_REPEATS = 100
VERBOSITY = 4


def main():
    ds = data_loader.get_imagen_data_sets(file_c="mats_mid_bsl.mat",
                                          # y_col_c="csi_c_sum_fu2",
                                          as_r=False,
                                          clean_data=True)
    out_df = pd.DataFrame(columns=["descriptor", "mask_type", "spearman_rho"])
    for d in ds:
        p("Starting CPM for {}...".format(d.descriptor), 2, VERBOSITY)
        x = d.x
        y = d.y

        x = convert_to_wide(x)
        for mask_type in ["all", "positive", "negative"]:
            p("Calculating spearman rho for {}...".format(mask_type), 3, VERBOSITY)
            cpm_spearman = get_cpm_spearman_average(x, y, mask_type, n_repeats=N_REPEATS)
            out_df = out_df.append({"descriptor": d.get_descriptor(), "mask_type": mask_type,
                                    "spearman_rho": cpm_spearman},
                                   ignore_index=True)
    p("Saving results...", 1, VERBOSITY)
    out_df.to_csv("cpm_rho_values.csv")


def get_cpm_spearman_average(x, y, mask_type, n_repeats=1):

    rhos = np.ndarray((n_repeats,))
    for i in range(n_repeats):
        p("Calculating spearman rho repeat {}/{}...".format(i+1, n_repeats), 4, VERBOSITY)
        kf = KFold(n_splits=N_FOLDS, shuffle=True)
        y_hat = np.ndarray(y.shape[0])
        for split_index, (train_index, test_index) in enumerate(kf.split(x)):
            p("Fitting fold {}/{}...".format(split_index+1, N_FOLDS), 5, VERBOSITY)
            reg, masker = fit(x[train_index], y[train_index], mask_type)
            x_test, n_params = masker.get_x(CPM_THRESHOLD, mask_type, x[test_index])
            y_hat[test_index] = reg.predict(x_test)
        rhos[i] = spearmanr(y, y_hat)[0]
        p("Spearman rho: {}".format(rhos[i]), 4, VERBOSITY)
    ret = np.mean(rhos)
    p("Average spearman rho: {}".format(ret), 3, VERBOSITY)
    return ret


def fit(x, y, mask_type):
    masker = CPMMasker(x, y)
    xs, n_vals = masker.get_x(CPM_THRESHOLD, mask_type)
    return LinearRegression().fit(xs, y), masker


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
