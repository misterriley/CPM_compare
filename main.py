
import math
import multiprocessing

import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
from mpire import WorkerPool
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# MAT_FILES_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_mat/"
# MAT_FILES_PATH = "G:/My Drive/CPM_test_data"
MAT_FILES_PATH = ["G:/.shortcut-targets-by-id/1P67X2oPl5kWND4p5dhdn9RbtEZcHaiZ9/Data CPM 2460 Accuracy Interference",
                  "rest_estroop_acc_interf_2460_cpm_ready.mat"]
COLLAPSE_VECTORS = True
N_FOLDS = 5  # if None, leave-out-out CV is used with no randomization
N_REPEATS = 20
RANDOMIZE_DATA = True
N_JOBS = 7  # multiprocessing.cpu_count() - 2
N_ALPHAS = 61
DESCS = ["negative", "positive", "all"]
SCATTER_ALPHAS = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]


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
                ret = x_[:, [pos_mask[j] or neg_mask[j] for j in range(len(self.corrs))]]
            n_params = n_pos + n_neg

        return ret, n_params


def get_masker(fold_index, x_, y_, kf_):
    train_indices = kf_[fold_index][0]
    x_train, y_train = x_[train_indices], y_[train_indices]
    return CPMMasker(x_train, y_train)


def run_one_cpm(alpha, kf_, fold_masker_arr_, desc_, x_, y_, repeat_index_):
    y_pred = np.zeros(y_.shape)
    total_params = 0
    for j in range(len(kf_)):
        train_indices = kf_[j][0]
        test_indices = kf_[j][1]

        fold_masker = fold_masker_arr_[j]
        train_x_masked, n_params = fold_masker.get_x(alpha, desc_)
        train_y = y_[train_indices]

        model = LinearRegression()
        model.fit(train_x_masked, train_y)

        test_x, n_params = fold_masker.get_x(alpha, desc_, x_[test_indices])
        y_pred[test_indices] = model.predict(test_x)
        total_params += n_params

    print(f"\t{desc_} alpha {alpha:.6f} repeat {repeat_index_ + 1} finished")
    rho = stats.spearmanr(y_, y_pred)[0]
    avg_params = total_params / len(kf_)
    return alpha, rho, avg_params


def print_efficiency(insights_, n_jobs_):
    print(f"threads: {n_jobs_}, efficiency: {insights_['working_ratio'] * n_jobs_:.2f}")


def display_data(data_df_, log_x_, log_y_, folds_, repeats, desc_, source):
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
    plt.title(desc_)

    if log_x_:
        ax.set_xscale("log")
        ax2.set_xscale("log")
    if log_y_:
        ax.set_yscale("log")
        ax2.set_yscale("log")

    save_file_name = f"{source}_{desc_}_{folds_}fold_{repeats}_repeats" \
                     f"{'_logx' if log_x_ else ''}{'_logy' if log_y_ else ''}.png"

    plt.savefig(save_file_name)
    plt.close()


def do_one_repeat(shared_objects_, repeat_idx_):
    x_, y_, folds_, alphas_ = shared_objects_
    print(f"Started repeat {repeat_idx_ + 1}/{N_REPEATS}")
    kf = KFold(n_splits=folds_, shuffle=RANDOMIZE_DATA)
    kf = [f for f in kf.split(x_)]
    fold_masker_arr = [get_masker(j, x_, y_, kf) for j in range(folds_)]

    data_dfs_ = [None] * len(DESCS)
    for desc_index_ in range(len(DESCS)):
        cpm_results = [run_one_cpm(alpha, kf, fold_masker_arr, DESCS[desc_index_], x_, y_, repeat_idx_)
                       for alpha in alphas_]
        cpm_df = pd.DataFrame(cpm_results, columns=["alpha", "rho", "n_params"])
        data_dfs_[desc_index_] = pd.concat([data_dfs_[desc_index_], cpm_df])

    print(f"Finished repeat {repeat_idx_ + 1}/{N_REPEATS}")
    return data_dfs_


def load_data():
    print(f"Loading {MAT_FILES_PATH[1]}")
    mat_file = sio.loadmat(os.path.join(MAT_FILES_PATH[0], MAT_FILES_PATH[1]))

    if MAT_FILES_PATH[0] == "G:/My Drive/CPM_test_data":
        y_ = pd.read_csv(os.path.join(MAT_FILES_PATH[0], "txnegop.txt"), sep="\t", header=None)
        y_ = y_.values.reshape(-1)
        x_ = mat_file["stp_all"]
    else:
        y_ = mat_file["y"].reshape(-1)
        x_ = mat_file["x"]

    return x_, y_


if __name__ == '__main__':
    i = 0
    thread_counts = []
    last_thread_count = None
    while True:
        i += 1
        thread_count = int(math.ceil(N_REPEATS/i))
        if thread_count <= multiprocessing.cpu_count():
            if last_thread_count is None or last_thread_count - thread_count > 1:
                thread_counts.append(str(thread_count))
                last_thread_count = thread_count
            else:
                break

    print(f"recommended values for N_JOBS: {', '.join(thread_counts)}, or fewer")

    x, y = load_data()

    num_peeps = x.shape[2]
    num_nodes = x.shape[1]
    utix = np.triu_indices(num_nodes, k=1)
    x = x[utix[0], utix[1], :].transpose()

    job_count = N_JOBS if N_JOBS is not None else multiprocessing.cpu_count()
    folds = N_FOLDS if N_FOLDS is not None else num_peeps
    alphas = np.logspace(0, -6, num=N_ALPHAS, base=10)

    shared_objects = (x, y, folds, alphas)

    with WorkerPool(n_jobs=job_count, enable_insights=True, shared_objects=shared_objects) as pool:
        repeats_data = pool.map(do_one_repeat, range(N_REPEATS))
        print_efficiency(pool.get_insights(), job_count)

    for desc_index in range(len(DESCS)):
        grouped_data = pd.concat([repeats_data[i][desc_index] for i in range(len(repeats_data))],
                                 ignore_index=True)
        grouped_data = grouped_data.groupby(by=["alpha"]).mean()
        for log_x, log_y in ((True, True), (True, False), (False, True), (False, False)):
            display_data(grouped_data,
                         log_x,
                         log_y,
                         folds,
                         N_REPEATS,
                         DESCS[desc_index],
                         MAT_FILES_PATH[1].replace(".mat", ""))
