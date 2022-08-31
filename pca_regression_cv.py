import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import data_loader
import numpy as np
from main import convert_to_wide
import seaborn as sns

N_SPLITS = 10
MAX_PCS = 100


def get_regression(num_pcs, x_pca, y):
    x_to_fit = x_pca[:, :num_pcs]
    reg = LinearRegression()
    reg.fit(x_to_fit, y)
    return reg


def test_pcas(x_train, y_train, x_test, y_test, n_pcs):
    pca = PCA()
    x_pca_train = pca.fit_transform(x_train)
    x_pca_test = pca.transform(x_test)

    pca_y_corrs = [(i, pearsonr(x_pca_train[:, i], y_train)) for i in range(x_pca_train.shape[1])]
    pca_y_corrs = sorted(pca_y_corrs, key=lambda x: x[1][0])

    ret = np.zeros((n_pcs + 1, y_test.shape[0]))
    ret[0, :] = y_test.mean()
    for i in range(1, n_pcs + 1):
        pca_indices = [x[0] for x in pca_y_corrs[:i]]
        reg = LinearRegression()
        reg.fit(x_pca_train[:, pca_indices], y_train)
        y_pred = reg.predict(x_pca_test[:, pca_indices])
        ret[i] = y_pred

    return ret


def count_max_pcs(x):
    return min(x.shape[0], x.shape[1])


def main():
    ds = data_loader.get_test_data_sets(  # file_c="mats_mid_bsl.mat",
                                          # y_col_c="kirby_c_estimated_k_all_trials_fu2",
                                          as_r=True,
                                          clean_data=True)
    for d in ds:
        x = d.x
        y = d.y

        x = convert_to_wide(x)
        max_pcs = min(count_max_pcs(x), MAX_PCS)
        kf = KFold(n_splits=N_SPLITS, shuffle=True)
        y_preds = np.zeros((max_pcs + 1, y.shape[0]))
        for i, (train_index, test_index) in enumerate(kf.split(x)):
            print("Fold {}/{}".format(i + 1, kf.n_splits))
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            y_preds[:, test_index] = test_pcas(x_train, y_train, x_test, y_test, max_pcs)

        plt_df = pd.DataFrame()
        plt_df["n_pcs"] = np.arange(max_pcs + 1)
        plt_df["spearman_r"] = [spearmanr(y_preds[i], y)[0] for i in range(max_pcs + 1)]
        sns.scatterplot(plt_df["n_pcs"], plt_df["spearman_r"], legend="auto")
        plt.savefig("pca_cpm_{}.png".format(d.get_descriptor()))
        plt.close()

        plt_df["pearson_r"] = [pearsonr(y_preds[i], y)[0] for i in range(max_pcs + 1)]
        plt_df.to_csv("pca_cpm_{}.csv".format(d.get_descriptor()))


if __name__ == '__main__':
    main()
