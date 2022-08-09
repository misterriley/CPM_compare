import multiprocessing

import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
from mpire import WorkerPool
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# MAT_FILES_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_mat/"
MAT_FILES_PATH = "G:/My Drive/CPM_test_data"
COLLAPSE_VECTORS = True
N_FOLDS = 5  # if None, leave-out-out CV is used with no randomization
N_REPEATS = 100
RANDOMIZE_DATA = True
N_JOBS = 5
RUN_COMPARISON = False
N_ALPHAS = 201
DESCS = ["negative", "positive", "all"]


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
        n_pos = pos_masked.shape[1]
        n_neg = neg_masked.shape[1]

        if COLLAPSE_VECTORS:
            pos_masked = pos_masked.sum(axis=1).reshape(-1, 1)
            neg_masked = neg_masked.sum(axis=1).reshape(-1, 1)

        if mask_type == "positive":
            ret = pos_masked
            n_params = n_pos
        elif mask_type == "negative":
            ret = neg_masked
            n_params = n_neg
        else:
            if COLLAPSE_VECTORS:
                ret = pos_masked - neg_masked
            else:
                ret = x_[:, [pos_mask[i] or neg_mask[i] for i in range(len(self.corrs))]]
            n_params = n_pos + n_neg

        return ret, n_params


def run_lasso_comparison(x_, y_, cv=5, n_jobs_=-1):
    x_normed = (x_ - np.mean(x_, axis=0)) / np.std(x_, axis=0)

    for i in np.logspace(-3.5, -1.5, num=101, base=10):
        model = Lasso(alpha=i)
        scores = cross_val_score(model, x_normed, y_, cv=cv, n_jobs=n_jobs_)
        print(f"alpha = {i:.4f}: {scores.mean():.4f}")
        model.fit(x_normed, y_)
        non_zero_betas = np.count_nonzero(model.coef_)
        print(f"non_zero_betas = {non_zero_betas}")


def get_masker(fold_index, shared_objects_):
    kf_ = shared_objects_["kf"]
    x_ = shared_objects_["x"]
    y_ = shared_objects_["y"]

    train_indices = kf_[fold_index][0]
    x_train, y_train = x_[train_indices], y_[train_indices]
    return CPMMasker(x_train, y_train)


def run_one_cpm(alpha, shared_objects_):
    kf_ = shared_objects_["kf"]
    fold_masker_arr_ = shared_objects_["fold_masker_arr"]
    desc_ = shared_objects_["desc"]
    y_ = shared_objects_["y"]
    x_ = shared_objects_["x"]
    y_pred = np.zeros(y_.shape)
    total_params = 0
    for i in range(len(kf_)):
        train_indices = kf_[i][0]
        test_indices = kf_[i][1]

        fold_masker = fold_masker_arr_[i].clone()
        train_x_masked, n_params = fold_masker.get_x(alpha, desc_)
        train_y = y_[train_indices]
        model = LinearRegression()
        model.fit(train_x_masked, train_y)
        test_x, n_params = fold_masker.get_x(alpha, desc_, x_[test_indices])
        y_pred[test_indices] = model.predict(test_x)
        total_params += n_params
    rho = stats.spearmanr(y_, y_pred)[0]
    avg_params = total_params / len(kf_)
    # print(f"alpha = {alpha:.8f}: desc = {desc_}: rho = {rho:.4f}: n_params = {avg_params:.4f}")
    return alpha, rho, avg_params


def print_efficiency(insights_, n_jobs_):
    print(f"threads: {n_jobs_}, efficiency: {insights_['working_ratio'] * n_jobs_:.2f}")


def display_data(data_df_, log_x_, log_y_, folds_, repeats):
    ax = sns.lineplot(x="alpha",
                      y="rho",
                      data=data_df_,
                      label="rho",
                      legend=False)
    ax2 = ax.twinx()
    sns.lineplot(x="alpha",
                 y="n_params",
                 data=data_df_,
                 ax=ax2,
                 label="n_params",
                 legend=False,
                 color="red")
    ax.figure.legend()
    plt.title(desc)

    if log_x_:
        ax.set_xscale("log")
        ax2.set_xscale("log")
    if log_y_:
        ax.set_yscale("log")
        ax2.set_yscale("log")

    save_file_name = f"{desc}_{folds_}fold_{repeats}_repeats_{'_logx' if log_x_ else ''}{'_logy' if log_y_ else ''}.png"

    plt.savefig(save_file_name)
    plt.close()


if __name__ == '__main__':
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

        if RUN_COMPARISON:
            run_lasso_comparison(x, y, cv=len(y))

        job_count = N_JOBS if N_JOBS is not None else multiprocessing.cpu_count()
        folds = N_FOLDS if N_FOLDS is not None else num_peeps
        alphas = np.logspace(0, -6, num=N_ALPHAS, base=10)

        data_dfs = [pd.DataFrame(columns=["alpha", "rho", "n_params"]) for _ in range(len(DESCS))]
        with WorkerPool(n_jobs=job_count, enable_insights=True) as pool:
            for repeat_idx in range(N_REPEATS):
                print(f"Repeat {repeat_idx+1}/{N_REPEATS}")
                kf = KFold(n_splits=folds, shuffle=RANDOMIZE_DATA)
                shared_objects = {"x": x, "y": y, "kf": [f for f in kf.split(x)]}

                shared_objects["fold_masker_arr"] = pool.map(get_masker,
                                                             zip(range(folds), [shared_objects for _ in range(folds)]),
                                                             chunk_size=folds / job_count)

                for desc_index in range(len(DESCS)):
                    desc = DESCS[desc_index]
                    print(f"{desc}")
                    shared_objects["desc"] = desc
                    cpm_results = pool.map(run_one_cpm,
                                           zip(alphas, [shared_objects for i in alphas]),
                                           chunk_size=int(len(alphas) / job_count))

                    cpm_df = pd.DataFrame(cpm_results, columns=["alpha", "rho", "n_params"])
                    data_dfs[desc_index] = pd.concat([data_dfs[desc_index], cpm_df])

                print_efficiency(pool.get_insights(), job_count)

        for desc_index in range(len(DESCS)):
            grouped_data = data_dfs[desc_index]
            grouped_data = grouped_data[grouped_data["rho"] > 0].groupby(by=["alpha"])  # don't care about negative rho
            for log_x, log_y in ((True, True), (True, False), (False, True), (False, False)):
                display_data(grouped_data.mean(), log_x, log_y)
