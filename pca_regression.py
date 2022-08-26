import os.path
import time

import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from mpire import WorkerPool
from sklearn.preprocessing import StandardScaler

import data_loader
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Lasso

N_SPLITS = 10
MAX_PCAS = 45
N_JOBS = 4
N_REPEATS = 100
DO_PCA_REGRESSION = False
MAX_ITER = 100000


def predict_y(x_train, y_train, x_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred


# noinspection PyTypeChecker
def one_full_cv(shared_objects, param):
    x_, y_ = shared_objects
    n_pcs = l1_constant = lr = None
    if DO_PCA_REGRESSION:
        n_pcs = param
    else:
        l1_constant = param
        # if l1_constant == 0 and Lasso() is used then it throws warnings
        lr = LinearRegression() if l1_constant == 0 else \
            Lasso(l1_constant, warm_start=True, max_iter=MAX_ITER)

    kf = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle=True)
    y_preds = [None] * x_.shape[1]
    num_nonzero_coefs = [None] * x_.shape[1]
    fold_index = 0
    for train_index, test_index in kf.split(y_):
        fold_index += 1

        x_train = x_[:, train_index]
        x_test = x_[:, test_index]
        y_train = y_[train_index]
        nnc = None
        if DO_PCA_REGRESSION:
            y_pred = pca_regression(n_pcs, x_test, x_train, y_train)
        else:
            y_pred, nnc = lasso_regression(x_test, x_train, y_train, lr)

        for test_idx in range(len(test_index)):
            y_preds[test_index[test_idx]] = y_pred[test_idx]
            if not DO_PCA_REGRESSION:
                num_nonzero_coefs[test_index[test_idx]] = nnc
    y_preds = np.array(y_preds)

    pearson_r = pearsonr(y_preds, y_)[0]
    spearman_r = spearmanr(y_preds, y_)[0]
    mse = np.mean((y_preds - y_) ** 2)

    if DO_PCA_REGRESSION:
        print("\n --- {} pca vector(s) --- ".format(n_pcs))
    else:
        print("\n --- {} l1 constant --- ".format(l1_constant))
    print("pearson r ", pearson_r)
    print("spearman r", spearman_r)
    print("mse    ", mse)
    if not DO_PCA_REGRESSION:
        print("num nonzero coefs", np.mean(num_nonzero_coefs))

    if not DO_PCA_REGRESSION:
        return pearson_r, spearman_r, mse, np.mean(num_nonzero_coefs)
    return pearson_r, spearman_r, mse


def lasso_regression(x_test, x_train, y_train, lr):
    lr.fit(x_train.T, y_train)
    y_pred = lr.predict(x_test.T)
    nnc = np.sum(lr.coef_ != 0)
    return y_pred, nnc


# noinspection PyTypeChecker
def pca_regression(n_pcs, x_test, x_train, y_train):
    pca = PCA(n_components=MAX_PCAS)
    pca.fit(x_train.T)
    x_pca_train = pca.transform(x_train.T)
    x_pca_test = pca.transform(x_test.T)
    corrs = [None] * pca.components_.shape[0]
    for comp_index in range(len(corrs)):
        pca_vals = x_pca_train[:, comp_index]
        corrs[comp_index] = (comp_index, spearmanr(pca_vals, y_train))
    corrs = sorted(corrs, key=lambda z: z[1][1])
    x_train_best = x_pca_train[:, [corrs[i][0] for i in range(n_pcs)]]
    x_test_best = x_pca_test[:, [corrs[i][0] for i in range(n_pcs)]]
    y_pred = predict_y(x_train_best, y_train, x_test_best)
    return y_pred


def do_pca_regression(worker_pool, out_df):
    for n_pca_vectors in range(1, MAX_PCAS + 1):
        results = worker_pool.map(one_full_cv, [n_pca_vectors] * N_REPEATS)
        for i in range(len(results)):
            out_df.loc[len(out_df)] = [n_pca_vectors, i, *results[i]]


def do_l1_regression(worker_pool, out_df):
    for l1_const in [0, *np.logspace(-15, 7, num=MAX_PCAS)]:
        begin = time.time()
        results = worker_pool.map(one_full_cv, [l1_const] * N_REPEATS)
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
    plt.savefig("results/pca_regression_results_{}_{}.png".format(y_col, data_set.get_descriptor()))
    plt.close('all')


if __name__ == "__main__":
    ds = data_loader.get_test_data_sets(as_r=False, clean_data=True)
    for d in ds:
        x = d.get_x()
        y = d.get_y()

        triu = np.triu_indices(x.shape[1], k=1)
        x_triu = StandardScaler().fit_transform(x[triu[0], triu[1], :])

        if DO_PCA_REGRESSION:
            col_names = ["n_pcas", "repeat", "pearson_r", "spearman_r", "mse"]
        else:
            col_names = ["l1_constant", "repeat", "pearson_r", "spearman_r", "mse", "n_nonzero_coefs"]

        results_df = pd.DataFrame(columns=col_names)
        with WorkerPool(n_jobs=N_JOBS, shared_objects=(x_triu, y)) as pool:
            now = time.time()
            if DO_PCA_REGRESSION:
                do_pca_regression(pool, results_df)
            else:
                do_l1_regression(pool, results_df)
            elapsed = time.time() - now
            print("elapsed time: {}".format(elapsed))

        if not os.path.exists("results"):
            os.makedirs("results")
        results_df.to_csv("results/pca_regression_results_{}.csv".format(d.get_descriptor()))

        use_log_scale = not DO_PCA_REGRESSION
        if DO_PCA_REGRESSION:
            x_col = "n_pca_components"
        else:
            x_col = "l1_constant"

        grouped_results = results_df.groupby(x_col).mean().reset_index()

        save_one_item(grouped_results, x_col, "pearson_r", "Pearson correlation coefficient", d, use_log_scale)
        save_one_item(grouped_results, x_col, "spearman_r", "Spearman correlation coefficient", d, use_log_scale)
        save_one_item(grouped_results, x_col, "mse", "Mean Squared Error", d, use_log_scale)
        if "n_nonzero_coefs" in grouped_results.columns:
            save_one_item(grouped_results, x_col, "n_nonzero_coefs", "Average Coefficients in Model", d, use_log_scale)
