import numpy as np
from mpire import WorkerPool
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings

import data_loader
from main import CPMMasker, convert_to_wide

NUM_REPEATS = 100
NUM_FOLDS = 10
N_JOBS = None
CPM_THRESHOLD = 0.05

NUM_LAMBDAS = 21
MIN_LASSO_COEF = 10 ** -2
MAX_LASSO_COEF = 10 ** 2

CPM_ONLY = False
VERBOSITY = 1


def p(s, threshold):
    if VERBOSITY >= threshold:
        print(s)


def get_predictions(x, y, train_index, test_index, lasso_coefs, mask_type, cpm_only):
    if not type(lasso_coefs) == np.ndarray:
        lasso_coefs = np.array([lasso_coefs])

    x_train = x[train_index]
    y_train = y[train_index]

    masker = CPMMasker(x_train, y_train)

    if cpm_only:
        x_train, _ = masker.get_x(CPM_THRESHOLD, mask_type)
        x_test, _ = masker.get_x(CPM_THRESHOLD, mask_type, x_=x[test_index])
        lr = LinearRegression()
        lr.fit(x_train.reshape(-1, 1), y_train)
        return lr.predict(x_test.reshape(-1, 1)), masker.count_cpm_coefficients(CPM_THRESHOLD, mask_type)

    mask = masker.get_mask_by_type(CPM_THRESHOLD, mask_type, as_digits=False)
    x_train = x_train[:, mask]
    x_test = x[test_index][:, mask]
    ret = np.zeros((lasso_coefs.shape[0], x_test.shape[0]))
    reg = Lasso(alpha=0, max_iter=1e5, warm_start=True)
    n_coefs = 0
    for i, lasso_coef in enumerate(lasso_coefs):
        reg.alpha = lasso_coef
        if x_train.shape[1] == 0:
            ret[i, :] = y_train.mean()
        else:
            reg.fit(x_train, y_train)
            n_coefs += sum(reg.coef_ != 0)
            y_pred = reg.predict(x_test)
            ret[i, :] = y_pred
    return ret, n_coefs / lasso_coefs.shape[0]


def select_best_lambda(x_train, y_train, mask_type, cpm_only):
    if cpm_only:
        return None
    if NUM_LAMBDAS == 1:
        return MIN_LASSO_COEF

    x_tms = StandardScaler().fit_transform(x_train)
    y_tms = y_train - y_train.mean()

    lasso_kf = KFold(n_splits=NUM_FOLDS, shuffle=True)
    splits = list(lasso_kf.split(y_tms))
    lasso_coefs = np.logspace(np.log10(MIN_LASSO_COEF),
                              np.log10(MAX_LASSO_COEF),
                              NUM_LAMBDAS)

    preds = np.zeros((NUM_LAMBDAS, y_tms.shape[0]))
    for i, split in enumerate(splits):
        p("Inner Fold {}".format(i), 2)
        values, _ = get_predictions(x_tms, y_tms, split[0], split[1], lasso_coefs, mask_type, cpm_only)
        preds[:, split[1]] = values

    best_score = -np.inf
    best_lambda = None
    for i in range(NUM_LAMBDAS):
        statistic = spearmanr(preds[i], y_tms)
        if statistic.correlation > best_score:
            best_score = statistic.correlation
            best_lambda = lasso_coefs[i]
    p("Best Lasso Coef: {}, spearman_r: {}".format(best_lambda, best_score), 2)
    return best_lambda


def do_one_repeat(shared_objects, repeat_index):
    warnings.filterwarnings("ignore")
    x, y = shared_objects
    p("Repeat {}".format(repeat_index), 1)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True)
    preds = np.zeros(y.shape[0])
    mask_type = "all"
    for i, split in enumerate(kf.split(y)):
        p("Outer Fold {}".format(i), 2)
        train_index, test_index = split
        best_lambda = select_best_lambda(x[train_index], y[train_index], mask_type, CPM_ONLY)
        values, coefs = get_predictions(x, y, train_index, test_index, best_lambda, mask_type, CPM_ONLY)
        preds[test_index] = values
        if not CPM_ONLY:
            p("retained {:.2f} coefficient(s)".format(coefs), 2)
    sp = spearmanr(preds, y)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    p(" ".join([str(sp), str(rmse)]), 1)
    return sp, rmse


def main():
    ds = data_loader.get_test_data_sets(  # file_c="mats_mid_bsl.mat",
        as_r=False,
        clean_data=True)

    for data_set in ds:
        x = data_set.get_x()
        y = data_set.get_y()

        x = convert_to_wide(x)

        with WorkerPool(n_jobs=N_JOBS, shared_objects=(x, y)) as pool:
            results = pool.map(do_one_repeat, range(NUM_REPEATS))

        mean_spr = np.mean([r[0].correlation for r in results])
        mean_rmse = np.mean([r[1] for r in results])
        p("mean spearmanr: {}, mean rmse: {}".format(mean_spr, mean_rmse), 1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
