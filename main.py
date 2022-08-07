import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
from mpire import WorkerPool
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# MAT_FILES_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_mat/"
MAT_FILES_PATH = "G:/My Drive/CPM_test_data"
COLLAPSE_VECTORS = True


class CPMMasker:

    def __init__(self, x_, y_, corrs_=None):
        self.corrs = corrs_
        if self.corrs is None:
            self.corrs = [stats.pearsonr(a, y_) for a in x_.transpose()]
        self.x = x_
        self.y = y_

    def clone(self):
        return CPMMasker(self.x, self.y, self.corrs)

    def get_x(self, threshold, mask_type, x_=None):

        if x_ is None:
            x_ = self.x

        if len(x_.shape) == 1:
            x_ = x_.reshape(-1, 1)

        pos_mask = [c[1] < threshold and c[0] > 0 for c in self.corrs]
        neg_mask = [c[1] < threshold and c[0] < 0 for c in self.corrs]

        pos_masked = x_[:, pos_mask]
        neg_masked = x_[:, neg_mask]

        if COLLAPSE_VECTORS:
            pos_masked = pos_masked.sum(axis=1).reshape(-1, 1)
            neg_masked = neg_masked.sum(axis=1).reshape(-1, 1)

        if mask_type == "positive":
            ret = pos_masked
        elif mask_type == "negative":
            ret = neg_masked
        else:
            if COLLAPSE_VECTORS:
                ret = pos_masked - neg_masked
            else:
                ret = x_[:, [pos_mask[i] or neg_mask[i] for i in range(len(pos_mask))]]

        return ret


def run_lasso_comparison(x_, y_, cv=5, n_jobs=-1):

    x_normed = (x_ - np.mean(x_, axis=0)) / np.std(x_, axis=0)

    for i in np.logspace(-3.5, -1.5, num=101, base=10):
        model = Lasso(alpha=i)
        scores = cross_val_score(model, x_normed, y_, cv=cv, n_jobs=n_jobs)
        print(f"alpha = {i:.4f}: {scores.mean():.4f}")
        model.fit(x_normed, y_)
        non_zero_betas = np.count_nonzero(model.coef_)
        print(f"non_zero_betas = {non_zero_betas}")


def get_masker(x_, y_):
    return CPMMasker(x_, y_)


def run_one_cpm(shared_objects_, alpha, desc_):
    fold_masker_arr_, fold_y_arr_, x_, y_, num_peeps_ = shared_objects_
    y_pred = np.ndarray(shape=(len(y_),), dtype=np.float32)
    for i in range(num_peeps_):
        fold_masker = fold_masker_arr_[i].clone()
        fold_x_masked = fold_masker.get_x(alpha, desc_)
        fold_y = fold_y_arr_[i].reshape(-1, 1)
        model = LinearRegression()
        model.fit(fold_x_masked, fold_y.reshape(-1, 1))
        pred_x = fold_masker.get_x(alpha, desc_, x_[i].reshape(1, -1))
        y_pred[i] = model.predict(pred_x)[0]
    rho = stats.spearmanr(y_, y_pred)[0]
    print(f"alpha = {alpha:.10f}: desc = {desc_}: rho = {rho:.4f}")
    return alpha, rho


if __name__ == '__main__':

    import logging
    logging.captureWarnings(True)

    for file in os.listdir(MAT_FILES_PATH):

        if not file.endswith(".mat"):
            continue

        print(f"Loading {file}")
        mat_file = sio.loadmat(os.path.join(MAT_FILES_PATH, file))

        y = pd.read_csv(os.path.join(MAT_FILES_PATH, "txnegop.txt"), sep="\t", header=None)
        y = y.values.reshape(-1)

        x = mat_file["stp_all"]
        num_peeps = x.shape[2]
        num_nodes = x.shape[1]
        utix = np.triu_indices(num_nodes, k=1)
        x = x[utix[0], utix[1], :].transpose()

        RUN_COMPARISON = False
        if RUN_COMPARISON:
            run_lasso_comparison(x, y, cv=len(y))

        fold_x_arr = [np.concatenate((x[:i], x[i + 1:]), axis=0) for i in range(num_peeps)]
        fold_y_arr = [np.concatenate((y[:i], y[i + 1:]), axis=0) for i in range(num_peeps)]

        with WorkerPool(n_jobs=None) as pool:
            fold_masker_arr = pool.map(get_masker, zip(fold_x_arr, fold_y_arr))

        shared_objects = (fold_masker_arr, fold_y_arr, x, y, num_peeps)
        with WorkerPool(n_jobs=8, shared_objects=shared_objects) as pool:
            alphas = np.logspace(0, -20, num=1001, base=np.exp(1))

            for desc in ["all", "positive", "negative"]:
                cpm_results = pool.map(run_one_cpm, zip(alphas, [desc] * len(alphas)))

                data_df = pd.DataFrame(cpm_results, columns=["alpha", "rho"])
                data_df["log alpha"] = np.log10(data_df["alpha"])
                data_df = data_df[data_df["rho"] > 0]  # don't care about negative rho

                sns.lineplot(x="alpha", y="rho", data=data_df, label=desc)
                plt.savefig(f"{desc}_cv_plot.png")
                plt.show(block=False)
                plt.close()

                sns.lineplot(x="log alpha", y="rho", data=data_df, label=desc)
                plt.savefig(f"{desc}_cv_plot_log.png")
                plt.show(block=False)
                plt.close()
