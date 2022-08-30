import os.path
import time

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from mpire import WorkerPool
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

import data_loader

N_SPLITS = 10
MAX_PCAS = 45
NUM_L1_CONSTANTS = 1
N_JOBS = 1  # 8 uses all cores and freezes output on test data
N_REPEATS = 1
MAX_ITER = 10


def predict_y(x_train, y_train, x_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred


# noinspection PyTypeChecker
def one_full_cv(shared_objects, l1_constant, repeat_index=None):
    x_, y_ = shared_objects

    # if l1_constant == 0 and Lasso() is used then it throws warnings
    lr = LinearRegression() if l1_constant == 0 else \
        Lasso(l1_constant, warm_start=True, max_iter=MAX_ITER, precompute=True, selection='random')

    kf = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle=True)
    y_preds = [None] * x_.shape[1]
    num_nonzero_coefs = [None] * x_.shape[1]
    fold_index = 0
    for train_index, test_index in kf.split(y_):
        fold_index += 1
        print("repeat {} l1_const {:.2E} fold {} starting".format(repeat_index, float(l1_constant), fold_index))

        x_train = x_[:, train_index]
        x_test = x_[:, test_index]
        y_train = y_[train_index]
        y_pred, nnc = lasso_regression(x_test, x_train, y_train, lr)

        for test_idx in range(len(test_index)):
            y_preds[test_index[test_idx]] = y_pred[test_idx]
            num_nonzero_coefs[test_index[test_idx]] = nnc
    y_preds = np.array(y_preds)

    pearson_r = pearsonr(y_preds, y_)[0]
    spearman_r = spearmanr(y_preds, y_)[0]
    mse = np.mean((y_preds - y_) ** 2)

    print("\n --- {} l1 constant --- ".format(l1_constant))
    if repeat_index is not None:
        print("repeat {}".format(repeat_index))
    print("pearson r ", pearson_r)
    print("spearman r", spearman_r)
    print("mse    ", mse)
    print("num nonzero coefs", np.mean(num_nonzero_coefs))
    print()

    return pearson_r, spearman_r, mse, np.mean(num_nonzero_coefs)


def lasso_regression(x_test, x_train, y_train, lr):
    lr.fit(x_train.T, y_train)
    y_pred = lr.predict(x_test.T)
    nnc = np.sum(lr.coef_ != 0)
    return y_pred, nnc


def do_l1_regression(worker_pool, out_df):
    for l1_const in [-2.5]:  # *np.logspace(-2.5, -2.5, num=NUM_L1_CONSTANTS)]:
        begin = time.time()
        results = worker_pool.map(one_full_cv, [(l1_const, i) for i in range(N_REPEATS)])
        if l1_const == 0:
            l1_const = np.finfo(float).eps  # smallest possible value, must be positive since displayed log scale
        for i in range(len(results)):
            out_df.loc[len(out_df), :] = [l1_const, i, *results[i]]
        print("time elapsed: ", time.time() - begin)


def save_one_item(data, x_column, y_col, title, data_set, log_scale):
    plt.plot(x_column, y_col, data=data, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_col)
    if log_scale:
        plt.xscale("log")
    plt.savefig("results/l1_results_{}_{}.png".format(y_col, data_set.get_descriptor()))
    plt.close('all')


if __name__ == "__main__":
    ds = data_loader.get_imagen_data_sets(file_c="mats_sst_fu2.mat", as_r=False, clean_data=True)
    for d in ds:
        x = d.get_x()
        y = d.get_y()

        triu = np.triu_indices(x.shape[1], k=1)
        x_triu = StandardScaler().fit_transform(x[triu[0], triu[1], :])

        results_df = pd.DataFrame(columns=["l1_constant", "repeat", "pearson_r", "spearman_r", "mse", "n_nonzero_coefs"])
        with WorkerPool(n_jobs=N_JOBS, shared_objects=(x_triu, y)) as pool:
            now = time.time()
            do_l1_regression(pool, results_df)
            elapsed = time.time() - now
            print("elapsed time: {}".format(elapsed))

        if not os.path.exists("results"):
            os.makedirs("results")
        results_df.to_csv("results/pca_regression_results_{}.csv".format(d.get_descriptor()))

        use_log_scale = True
        x_col = "l1_constant"

        grouped_results = results_df.groupby(x_col).mean().reset_index()

        save_one_item(grouped_results, x_col, "pearson_r", "Pearson correlation coefficient", d, use_log_scale)
        save_one_item(grouped_results, x_col, "spearman_r", "Spearman correlation coefficient", d, use_log_scale)
        save_one_item(grouped_results, x_col, "mse", "Mean Squared Error", d, use_log_scale)
        if "n_nonzero_coefs" in grouped_results.columns:
            save_one_item(grouped_results, x_col, "n_nonzero_coefs", "Average Coefficients in Model", d, use_log_scale)
