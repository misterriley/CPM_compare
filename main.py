import math
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn
import seaborn as sns
from mpire import WorkerPool
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import data_loader

USE_TEST_DATA = True

DO_NESTED_KFOLD = False
N_OUTER_FOLDS = 3
N_OUTER_REPEATS = 15

N_FOLDS = 10  # if None, set to leave-one-out CV
N_REPEATS = 100
RANDOMIZE_DATA = True
if USE_TEST_DATA:
    N_JOBS = 5
else:
    N_JOBS = 3
N_ALPHAS = 0
DESCS = ["negative", "positive", "all"]
GUARANTEED_ALPHAS = [0.05]


class CPMMasker:

    def __init__(self, x_, y_, corrs_=None):
        self.corrs = corrs_
        if self.corrs is None:
            y_corr = (y_ - y_.mean()) / np.linalg.norm(y_)
            x_corr = (x_ - x_.mean(axis=0).reshape(1, -1)) / np.linalg.norm(x_, axis=0).reshape(1, -1)
            x_corr = np.nan_to_num(x_corr)
            self.corrs = x_corr.T @ y_corr
        self.x = x_
        self.y = y_

    def clone(self):
        return CPMMasker(self.x, self.y, self.corrs)

    def critical_r(self, alpha):
        dof = self.y.shape[0] - 2
        critical_t = stats.t.ppf(1 - alpha/2, dof)
        return critical_t / np.sqrt(dof + critical_t**2)

    def get_mask_by_type(self, threshold, mask_type, as_digits=False):
        if mask_type == "positive" or mask_type == "pos" or mask_type == "p":
            return self.get_pos_mask(threshold, as_digits)
        elif mask_type == "negative" or mask_type == "neg" or mask_type == "n":
            return self.get_neg_mask(threshold, as_digits)
        else:
            return self.get_all_mask(threshold, as_digits)

    def get_pos_mask(self, threshold=0.05, as_ones=False):
        critical_r = self.critical_r(threshold)
        if as_ones:
            ret = np.ones(self.corrs.shape)
            ret[self.corrs < critical_r] = 0
        else:
            ret = self.corrs > critical_r
        return ret

    def sort_key(self, i, mask_type):
        corr = self.corrs[i]
        if mask_type == "positive":
            return corr
        elif mask_type == "negative":
            return -corr
        else:
            return abs(corr)

    def get_coef_order(self, mask_type):
        return sorted(range(self.corrs.shape[0]),
                      key=lambda x: self.sort_key(x, mask_type), reverse=True)

    def get_neg_mask(self, threshold=0.05, as_neg_ones=False):
        critical_r = self.critical_r(threshold)
        if as_neg_ones:
            ret = -1 * np.ones(self.corrs.shape)
            ret[self.corrs > -critical_r] = 0
        else:
            ret = self.corrs < -critical_r
        return ret

    def get_all_mask(self, threshold=0.05, as_digits=False):
        critical_r = self.critical_r(threshold)
        if as_digits:
            ret = np.zeros(self.corrs.shape)
            ret[self.corrs > critical_r] = 1
            ret[self.corrs < -critical_r] = -1
        else:
            ret = np.absolute(self.corrs) > critical_r
        return ret

    def count_cpm_coefficients(self, threshold, mask_type):
        mask = self.get_mask_by_type(threshold, mask_type)
        return np.count_nonzero(mask)

    def get_x(self, threshold, mask_type, x_=None):

        if x_ is None:
            x_ = self.x

        if type(x_) is list:
            x_ = np.ndarray(x_)

        if len(x_.shape) == 1:
            x_ = x_.reshape(-1, 1)

        pos_mask = self.get_pos_mask(threshold, as_ones=False)
        neg_mask = self.get_neg_mask(threshold, as_neg_ones=False)

        pos_masked = x_[:, pos_mask]
        neg_masked = x_[:, neg_mask]
        n_pos = pos_masked.shape[1]
        n_neg = neg_masked.shape[1]

        pos_masked = pos_masked.sum(axis=1).reshape(-1, 1)
        neg_masked = neg_masked.sum(axis=1).reshape(-1, 1)

        if mask_type == "positive":
            ret = pos_masked
            n_params = n_pos
        elif mask_type == "negative":
            ret = neg_masked
            n_params = n_neg
        else:
            ret = pos_masked - neg_masked
            n_params = n_pos + n_neg

        return ret, n_params


def get_masker(fold_index, x_, y_, kf_):
    train_indices = kf_[fold_index][0]
    x_train, y_train = x_[train_indices], y_[train_indices]
    return CPMMasker(x_train, y_train)


def run_one_cpm(alpha_, kf_, fold_masker_arr_, desc_, x_, y_, repeat_index_):
    y_pred = np.zeros(y_.shape)
    total_params = 0
    for j in range(len(kf_)):
        train_indices = kf_[j][0]
        test_indices = kf_[j][1]

        fold_masker = fold_masker_arr_[j]
        train_x_masked, n_params = fold_masker.get_x(alpha_, desc_)
        train_y = y_[train_indices]

        model = LinearRegression()
        model.fit(train_x_masked, train_y)

        test_x, n_params = fold_masker.get_x(alpha_, desc_, x_[test_indices])
        y_pred[test_indices] = model.predict(test_x)
        total_params += n_params

    print(f"\t{desc_} alpha {alpha_:.6f} repeat {repeat_index_ + 1} finished")
    # noinspection PyTypeChecker
    rho = stats.spearmanr(y_, y_pred)[0]
    avg_params = total_params / len(kf_)
    return alpha_, rho, avg_params


def print_efficiency(insights_, n_jobs_):
    print(f"threads: {n_jobs_}, efficiency: {insights_['working_ratio'] * n_jobs_:.2f}")


def save_data(data_df_, folds_, repeats, desc_, source):
    ax = sns.lineplot(x="alpha",
                      y="rho",
                      data=data_df_,
                      label="rho",
                      color="blue")
    plt.legend()
    plt.title(source + " " + desc_)

    ax.set_xscale("log")

    save_file_name = f"{source}_{desc_}_{folds_}fold_{repeats}_repeats.png"

    save_file_dir = f"./outputs/{source}/"
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    plt.savefig(os.path.join(save_file_dir, save_file_name), bbox_inches="tight")
    plt.close("all")

    report_file_name = f"{source}_{desc_}_{folds_}fold_{repeats}_repeats.csv"
    data_df_.to_csv(os.path.join(save_file_dir, report_file_name))


def do_one_repeat(shared_objects_, repeat_idx_):
    x_, y_, folds_, alphas_, descs = shared_objects_
    print(f"Started repeat {repeat_idx_ + 1}")
    kf = KFold(n_splits=folds_, shuffle=RANDOMIZE_DATA)
    kf = [f for f in kf.split(x_)]
    print(f"Getting maskers ready")
    fold_masker_arr = [get_masker(j, x_, y_, kf) for j in range(folds_)]

    data_dfs_ = {}
    for desc in descs:
        cpm_results = [run_one_cpm(alpha_, kf, fold_masker_arr, desc, x_, y_, repeat_idx_)
                       for alpha_ in alphas_]
        cpm_df = pd.DataFrame(cpm_results, columns=["alpha", "rho", "n_params"])
        data_dfs_[desc] = cpm_df

    print(f"Finished repeat {repeat_idx_ + 1}")
    return data_dfs_


def print_recommended_thread_counts():
    thread_counts = set([int(math.ceil(N_REPEATS / i)) for i in range(1, N_REPEATS + 1)])
    thread_counts_str = [str(i) for i in sorted(thread_counts, reverse=True) if i <= multiprocessing.cpu_count()]

    print(f"recommended values for N_JOBS: {', '.join(sorted(thread_counts_str))}")

    global N_JOBS
    if N_JOBS is None:
        N_JOBS = max([x for x in thread_counts if x <= multiprocessing.cpu_count()])
        print(f"N_JOBS set to {N_JOBS}")


def do_search(x, y, folds, alphas, job_count, descriptor, save):
    shared_objects = (x, y, folds, alphas, DESCS)

    print("starting WorkerPool(n={})".format(job_count))
    with WorkerPool(n_jobs=job_count, enable_insights=True, shared_objects=shared_objects) as pool:
        repeats_data = pool.map(do_one_repeat, range(N_REPEATS))
        print_efficiency(pool.get_insights(), job_count)

    if save:
        for desc in DESCS:
            grouped_data = pd.concat([repeats_data[i][desc] for i in range(len(repeats_data))],
                                     ignore_index=True)
            grouped_data = grouped_data.groupby(by=["alpha"]).mean()

            save_data(grouped_data,
                      folds,
                      N_REPEATS,
                      desc,
                      descriptor)

    return repeats_data


def convert_to_wide(matrix_tensor):
    num_nodes = matrix_tensor.shape[1]
    utix = np.triu_indices(num_nodes, k=1)
    return matrix_tensor[utix[0], utix[1], :].transpose()


def main():
    print_recommended_thread_counts()

    if USE_TEST_DATA:
        sets = data_loader.get_test_data_sets(as_r=False, clean_data=True)
    else:
        sets = data_loader.get_imagen_data_sets(as_r=False, clean_data=True, file_c="mats_mid_bsl.mat")
    for data_set in sets:
        x = data_set.get_x()
        y = data_set.get_y()
        descriptor = data_set.get_descriptor()
        print("starting data set:", descriptor)

        num_peeps = x.shape[2]
        num_nodes = x.shape[1]

        x = convert_to_wide(x)

        job_count = N_JOBS if N_JOBS is not None else multiprocessing.cpu_count()
        folds = N_FOLDS if N_FOLDS is not None else num_peeps
        alphas = np.logspace(0, -4, num=N_ALPHAS, base=10)
        for alpha in GUARANTEED_ALPHAS:
            if alpha not in alphas:
                alphas = np.append(alphas, alpha)
        alphas = np.sort(alphas)

        do_search(x, y, folds, alphas, job_count, descriptor, True)


def analyze_outputs():
    analysis_df = pd.DataFrame(columns=["source", "desc", "alpha", "rho", "n_params", "note"])
    for root, dirs, files in os.walk("./transfer/outputs"):
        for f in files:
            desc = f.split("_")[-4]
            if f.endswith(".csv"):
                in_df = pd.read_csv(os.path.join(root, f))
                out_df = pd.DataFrame(columns=analysis_df.columns)
                max_rho = in_df["rho"].max()
                max_rho_index = in_df.loc[in_df["rho"] == max_rho].index.values[0]
                out_df.loc[0] = in_df.loc[max_rho_index]
                out_df.loc[0, "note"] = "max rho"

                for i in range(len(GUARANTEED_ALPHAS)):
                    alpha = GUARANTEED_ALPHAS[i]
                    alpha_index = in_df[in_df["alpha"] == alpha].index.values[0]
                    out_df.loc[i + 1, list(in_df.columns)] = in_df.loc[alpha_index, list(in_df.columns)].values
                    out_df.loc[i + 1, "note"] = "fixed alpha {}".format(alpha)

                    diff_index = i + 1 + len(GUARANTEED_ALPHAS)
                    rho_diff = out_df.loc[i + 1, "rho"] - max_rho
                    out_df.loc[diff_index, list(in_df.columns)] = in_df.loc[alpha_index, list(in_df.columns)].values
                    out_df.loc[diff_index, "rho"] = rho_diff
                    out_df.loc[diff_index, "note"] = "max rho - alpha = {} rho".format(alpha)

                out_df["source"] = root.split("\\")[-1]
                out_df["desc"] = desc
                analysis_df = pd.concat([analysis_df, out_df], ignore_index=True)
        analysis_df.to_csv("./transfer/analysis.csv", index=False)

    for desc in DESCS:
        chart_df = analysis_df[analysis_df["desc"] == desc]
        chart_df = chart_df.sort_values(by=["note"])
        seaborn.barplot(x="note", y="rho", data=chart_df, ci=95)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Condition")
        # plt.show(block=False)
        plt.savefig(f"./transfer/{desc}_rho.png", bbox_inches="tight")
        plt.close()

    summary_df = analysis_df.groupby(by=["desc", "note"]).mean()
    standard_errors = analysis_df.groupby(by=["desc", "note"]).apply(lambda x: x.sem()).reset_index()
    for row in standard_errors.itertuples():
        summary_df.loc[(row.desc, row.note), "se"] = row.rho

    summary_df = summary_df.reset_index()
    summary_df.to_csv("./transfer/summary.csv", index=False)


if __name__ == '__main__':
    main()
    # analyze_outputs()
