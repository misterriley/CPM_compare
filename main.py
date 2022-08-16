import math
import multiprocessing
import os

import scipy.stats as stats
import numpy as np
import seaborn
from mpire import WorkerPool
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import DataLoader

DO_NESTED_KFOLD = True
N_OUTER_FOLDS = 2
N_OUTER_REPEATS = 20
COLLAPSE_VECTORS = True
N_FOLDS = 5  # if None, set to leave-one-out CV
N_REPEATS = 20
RANDOMIZE_DATA = True
N_JOBS = 5
N_ALPHAS = 0
DESCS = ["negative", "positive", "all"]
GUARANTEED_ALPHAS = [0.1, 0.05, 0.01, 0.005, 0.001]


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
    plt.title(source + " " + desc_)

    ax.set_xscale("log")
    ax2.set_xscale("log")
    # ax.set_yscale("log")
    ax2.set_yscale("log")

    save_file_name = f"{source}_{desc_}_{folds_}fold_{repeats}_repeats.png"

    save_file_dir = f"./outputs/{source}/"
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    plt.savefig(os.path.join(save_file_dir, save_file_name), bbox_inches="tight")
    plt.close()

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
    thread_counts = [str(i) for i in sorted(thread_counts, reverse=True) if i <= multiprocessing.cpu_count()]

    print(f"recommended values for N_JOBS: {', '.join(sorted(thread_counts))}")


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


def do_nested_kfold(x, y, folds, alphas, job_count, descriptor):
    output = pd.DataFrame(columns=["alpha", "rho", "n_params", "desc"])
    for repeat_index in range(N_OUTER_REPEATS):
        kf_outer = KFold(n_splits=N_OUTER_FOLDS, shuffle=RANDOMIZE_DATA)
        for fold_index in range(N_OUTER_FOLDS):
            print(f" --- Started outer fold {fold_index + 1} in repeat {repeat_index + 1} --- ")
            f = list(kf_outer.split(x))[fold_index]
            results = pd.DataFrame(columns=["alpha", "rho", "n_params", "desc"])
            result = do_search(x[f[0], :], y[f[0]], folds, alphas, job_count, descriptor, False)
            for result_index in range(len(result)):
                for desc in DESCS:
                    result_df = result[result_index][desc]
                    result_df["desc"] = desc
                    results = pd.concat([results, result_df], ignore_index=True)

            results = results.groupby(by=["desc", "alpha"]).mean().reset_index()
            test_x = x[f[1], :]
            test_y = y[f[1]]
            print(f" --- Testing outer fold {fold_index + 1} --- ")
            for desc in DESCS:
                print(f" --- Testing {desc} --- ")
                best_rho_idx = results.loc[results["desc"] == desc, "rho"].idxmax()
                best_alpha = results.loc[best_rho_idx, "alpha"]

                test_outcome = do_one_repeat((test_x, test_y, folds, [best_alpha], [desc]), 0)
                test_df = test_outcome[desc]
                test_df["desc"] = desc
                output = pd.concat([output, test_df], ignore_index=True)

    if not os.path.exists(f"./outputs/{descriptor}"):
        os.makedirs(f"./outputs/{descriptor}")
    output.to_csv(f"./outputs/{descriptor}/{descriptor}_{folds}fold_{N_OUTER_FOLDS}_outer_folds.csv")


def main():
    print_recommended_thread_counts()

    dl = DataLoader(
        protocol_c=[
            # "IMAGEN",
            "sadie-marie",
            # "test_data"
        ],
        file_c=[
            # "mats_sst_fu2.mat",
            # "rest_estroop_acc_interf_2460_cpm_ready.mat",
            "enback_estroop_acc_interf_2460_cpm_ready.mat",
            # "stp_all_clean2.mat"
        ],
        y_col_c=None)

    for data_set in dl.get_data_sets():
        x = data_set.get_x()
        y = data_set.get_y()
        descriptor = data_set.get_descriptor()
        print("starting data set:", descriptor)

        print("initial x shape:", x.shape)
        print("initial y shape:", y.shape)

        assert x.shape[2] == y.shape[0]
        assert x.shape[0] == x.shape[1]

        x_is_bad = [np.isnan(x[:, :, i]).any() for i in range(x.shape[2])]
        y_is_bad = np.isnan(y)
        good_indices = np.where(~np.logical_or(x_is_bad, y_is_bad))[0]
        x = x[:, :, good_indices]
        y = y[good_indices]

        print("expurgated x shape:", x.shape)
        print("expurgated y shape:", y.shape)

        num_peeps = x.shape[2]
        num_nodes = x.shape[1]

        # x is symmetric - cut out the upper triangle
        utix = np.triu_indices(num_nodes, k=1)
        x = x[utix[0], utix[1], :].transpose()

        job_count = N_JOBS if N_JOBS is not None else multiprocessing.cpu_count()
        folds = N_FOLDS if N_FOLDS is not None else num_peeps
        alphas = np.logspace(0, -6, num=N_ALPHAS, base=10)
        for alpha in GUARANTEED_ALPHAS:
            if alpha not in alphas:
                alphas = np.append(alphas, alpha)
        alphas = np.sort(alphas)

        if DO_NESTED_KFOLD:
            do_nested_kfold(x, y, folds, alphas, job_count, descriptor)
        else:
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
