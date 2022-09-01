import numpy as np
import pandas as pd
import win32api
import win32con
import win32process
from mpire import WorkerPool
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import data_loader
from main import convert_to_wide, CPMMasker

N_REPEATS = 20
N_SPLITS = 10
MAX_ENTRIES = 500
NUM_CHECKS = 50
VERBOSITY = 4
N_JOBS = 7


def get_check_list(min_entries, max_entries, num_checks):
    assert min_entries > 0

    ret = np.ndarray(num_checks, dtype=int)
    ret[0] = last_entry = min_entries
    for i in range(1, num_checks):
        next_entry = (max_entries/last_entry)**(1/(num_checks-i)) * last_entry
        next_entry = int(round(next_entry))
        if next_entry == last_entry:
            next_entry += 1
        last_entry = next_entry
        ret[i] = last_entry

    return ret


def p(repeat_index, s, threshold, verbosity=VERBOSITY):
    if verbosity >= threshold:
        print("({}) {}".format(repeat_index, s))


def regress_on_best_entries(repeat_index, x_train, y_train, x_test, y_test, mask_type, n_entries=None):
    masker = CPMMasker(x_train, y_train)
    coefficient_order = masker.get_coef_order(mask_type)

    ret = None
    if n_entries is None:
        ret = np.ndarray(shape=(MAX_ENTRIES + 1, y_test.shape[0]), dtype=object)
        ret[0, :] = y_train.mean()

    if n_entries == 0:
        return np.repeat(y_train.mean(), y_test.shape[0])

    dim_range = get_check_list(1, MAX_ENTRIES, NUM_CHECKS) if n_entries is None else [n_entries]

    for i, dim in enumerate(dim_range):
        p(repeat_index, "Dimension = {} ({}/{})".format(dim, i + 1, NUM_CHECKS), 6)
        dof = x_train.shape[0] - dim - 1
        if dof < 0:
            break
        mask = np.zeros(x_train.shape[1], dtype=bool)
        mask[coefficient_order[:dim]] = True

        reg = LinearRegression()
        reg.fit(x_train[:, mask], y_train)
        y_hat = reg.predict(x_test[:, mask])

        if n_entries is None:
            ret[dim] = y_hat
        else:
            return y_hat

    return ret

def decrease_priority():

    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, win32process.THREAD_PRIORITY_LOWEST)


def main():
    decrease_priority()
    ds = data_loader.get_imagen_data_sets(file_c=["mats_mid_fu2.mat", "mats_sst_bsl.mat", "mats_sst_fu2.mat"],
                                          y_col_c=None,
                                          as_r=False,
                                          clean_data=True)
    for d in ds:
        x = d.x
        y = d.y

        x = convert_to_wide(x)
        p(-1, "descriptor: {}".format(d.descriptor), 1)
        for mask_type in ["all", "positive", "negative"]:
            p(-1, "Mask type: {}".format(mask_type), 2)
            with WorkerPool(N_JOBS, shared_objects=(x, y, mask_type)) as pool:
                results = np.array(pool.map_unordered(one_outer_repeat, range(1, N_REPEATS + 1)))
            out_df = pd.DataFrame({"run_index": np.arange(1, N_REPEATS + 1),
                                   "spearman r": results[:, 0],
                                   "best_dim": results[:, 1],
                                   "mask_type": mask_type})
            out_df.loc[len(out_df)] = ["average",
                                       out_df["spearman r"].mean(),
                                       out_df["best_dim"].mean(),
                                       mask_type]

            out_df.to_csv("C:\\Users\\bleem\\Dropbox\\Yale\\Output Sync\\s_and_r_{}_{}.csv".format(d.get_descriptor(), mask_type), index=False)
            p(-1, "{} {} {}".format(d.get_descriptor(), np.mean(results[:, 0]), np.mean(results[:, 1])), 2)


def one_outer_repeat(shared_objects, repeat_index):
    try:
        x, y, mask_type = shared_objects
        p(repeat_index, "Repeat {}/{}".format(repeat_index, N_REPEATS), 3)
        kf = KFold(n_splits=N_SPLITS, shuffle=True)
        folds = list(kf.split(x))
        y_hat = np.zeros(y.shape[0])
        for i, (train_index, test_index) in enumerate(folds):
            p(repeat_index, "Outer Fold {}/{}".format(i + 1, kf.n_splits), 4)
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            best_dim = get_best_dim(repeat_index, x_train, y_train, mask_type)
            y_hat[test_index] = regress_on_best_entries(repeat_index, x_train, y_train, x_test, y_test, mask_type,
                                                        best_dim)
        best_dim = get_best_dim(repeat_index, x, y, mask_type, folds)
        p(repeat_index, "Best dim: {}".format(best_dim), 3)
        return spearmanr(y_hat, y)[0], best_dim
    except Exception as e:
        print(e)
        exit()


def get_best_dim(repeat_index, x, y, mask_type, folds=None):
    if folds is None:
        folds = list(KFold(n_splits=N_SPLITS, shuffle=True).split(x))
    y_hat = np.zeros((MAX_ENTRIES + 1, y.shape[0]))
    for i, (train_index, test_index) in enumerate(folds):
        p(repeat_index, "Fold {}/{}".format(i + 1, len(folds)), 5)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_hat[:, test_index] = regress_on_best_entries(repeat_index, x_train, y_train, x_test, y_test, mask_type)
    spearman_correlations = [spearmanr(y_hat[i], y)[0] for i in range(MAX_ENTRIES + 1)]
    best = np.nanargmax(spearman_correlations)
    return best


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    main()
