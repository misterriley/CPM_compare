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
from sklearn.linear_model import LinearRegression

N_SPLITS = 10

MAX_PCAS = 45

N_JOBS = 3

N_REPEATS = 20


def predict_y(x_train, y_train, x_test):

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred


def one_full_cv(shared_objects, n_pcs):
    x_, y_ = shared_objects

    kf = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle=True)
    y_preds = [None] * x_.shape[1]
    for train_index, test_index in kf.split(y_):
        x_train = x_[:, train_index]
        x_test = x_[:, test_index]
        y_train = y_[train_index]

        pca = PCA(n_components = MAX_PCAS)
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
        for test_idx in range(len(test_index)):
            y_preds[test_index[test_idx]] = y_pred[test_idx]
    y_preds = np.array(y_preds)

    pearson_r = pearsonr(y_preds, y_)[0]
    spearman_r = spearmanr(y_preds, y_)[0]
    mse = np.mean((y_preds - y_) ** 2)

    print("\n --- {} pca vector(s) --- ".format(n_pcs))
    print("pearson r ", pearson_r)
    print("spearman r", spearman_r)
    print("mse    ", mse)

    return pearson_r, spearman_r, mse


if __name__ == "__main__":
    ds = data_loader.get_test_data_sets(as_r=False, clean_data=True)
    for d in ds:
        x = d.get_x()
        y = d.get_y()

        triu = np.triu_indices(x.shape[1], k=1)
        x_triu = StandardScaler().fit_transform(x[triu[0], triu[1], :])

        results_df = pd.DataFrame(columns=["n_pca_components", "rep", "pearson_r", "spearman_r",
                                           "mse"])
        with WorkerPool(n_jobs=N_JOBS, shared_objects=(x_triu, y)) as pool:
            now = time.time()
            for n_pca_vectors in range(1, MAX_PCAS + 1):
                results = pool.map(one_full_cv, [n_pca_vectors] * N_REPEATS)
                for i in range(len(results)):
                    results_df.loc[len(results_df)] = [n_pca_vectors, i, *results[i]]
            elapsed = time.time() - now
            print("elapsed time: {}".format(elapsed))

        if not os.path.exists("results"):
            os.makedirs("results")
        results_df.to_csv("results/pca_regression_results_{}.csv".format(d.get_descriptor()))

        grouped_results = results_df.groupby("n_pca_components").mean().reset_index()

        plt.plot("n_pca_components", "pearson_r", data=grouped_results, marker="o", linestyle="-", color="b")
        plt.title("Pearson correlation coefficient")
        plt.xlabel("n_pca_components")
        plt.ylabel("pearson_r")
        plt.savefig("results/pca_regression_results_{}_pearson_r.png".format(d.get_descriptor()))
        plt.close('all')

        plt.plot("n_pca_components", "spearman_r", data=grouped_results, marker="o", linestyle="-", color="b")
        plt.title("Spearman correlation coefficient")
        plt.xlabel("n_pca_components")
        plt.ylabel("spearman_r")
        plt.savefig("results/pca_regression_results_{}_spearman_r.png".format(d.get_descriptor()))
        plt.close('all')

        plt.plot("n_pca_components", "mse", data=grouped_results, marker="o", linestyle="-", color="b")
        plt.title("MSE")
        plt.xlabel("n_pca_components")
        plt.ylabel("mse")
        plt.savefig("results/pca_regression_results_{}_mse.png".format(d.get_descriptor()))
        plt.close('all')