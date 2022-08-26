import multiprocessing
import os
import time

import numpy as np
from mpire import WorkerPool
from scipy import stats
from sklearn.model_selection import KFold

import data_loader
import utils

NORM_NUMBER = .1


def dist_between_matrices(a, b):
    # https: // gmarti.gitlab.io / math / 2019 / 12 / 25 / riemannian - mean - correlation.html
    # _, eigenvals, _ = np.linalg.svd(np.linalg.solve(a, b))
    # return np.sqrt(np.sum(np.log(eigenvals) ** 2))

    # froebnius norm of the difference between the two matrices
    # return np.power(np.sum(np.power(np.abs(a - b), NORM_NUMBER)), 1 / NORM_NUMBER)
    return utils.wasserstein_distance_cov(a, b)


def get_distance(x_, i, j, verbose=False):
    if verbose:
        print(i, j)
    return i, j, dist_between_matrices(x_[:, :, i], x_[:, :, j])


def get_x_distance_matrix(x_data, description_data, n_jobs=3, read_from_file=True):
    save_file_name = "distance_matrix_{}.npy".format(description_data)
    if read_from_file and os.path.exists(save_file_name):
        return np.load(save_file_name)

    print("Starting distance matrix calculation")
    with WorkerPool(n_jobs=n_jobs, shared_objects=x_data) as pool:
        args = []
        for i in range(x_data.shape[2]):
            for j in range(i + 1, x_data.shape[2]):
                args.append((i, j))
        results = pool.map(get_distance, args)

    dists = np.zeros((x_data.shape[2], x_data.shape[2]))
    for i, j, dist in results:
        dists[i, j] = dist
        dists[j, i] = dist
    np.save(save_file_name, dists)
    return dists


def rbf(dist, h_):
    return np.cos(dist / h_) ** 2


def choose_h(x_data, y_data, description_data, read_from_file):
    best_h = 0
    best_r = -2
    dist_matrix = get_x_distance_matrix(x_data, description_data, read_from_file=read_from_file)
    y_diff_matrix = np.array([y_data] * y_data.shape[0]) - np.array([y_data] * y_data.shape[0]).T

    indices = np.triu_indices(dist_matrix.shape[0], k=1)

    x_vals = np.abs(dist_matrix[indices]).flatten()
    y_vals = np.abs(y_diff_matrix[indices]).flatten()
    # sns.scatterplot(x=x_vals, y=y_vals)
    a = np.sum(x_vals * y_vals) / np.sum(x_vals ** 2)
    errs = x_vals * a - y_vals
    ss_err = np.sum(errs ** 2)
    ss_t = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r_squared = 1 - (ss_err / ss_t)
    print("r_squared: {}".format(r_squared))
    # plt.show(block=True)

    for h_maybe in np.logspace(np.log10(.00001), np.log10(.01), num=10000, base=10):
        pred = rbf_reg(dist_matrix, h_maybe, y_data)
        if np.isnan(pred).any() or np.isinf(pred).any():
            continue
        r = stats.pearsonr(y_data, pred)
        print("h: {}, r: {}".format(h_maybe, r))
        if r[0] > best_r:
            best_h = h_maybe
            best_r = r[0]

    return best_h


def rbf_reg(dist_matrix, h_reg, y_data):
    kf = KFold(n_splits=y_data.shape[0]).split(y_data)
    pred = np.zeros(y_data.shape)
    for train_index, test_index in kf:
        for j in test_index:
            # add a tiny positive value to each weight to avoid division by zero
            weights = rbf(dist_matrix[j, train_index], h_reg) + 1e-10
            num = weights * y_data[train_index]
            pred[j] = np.sum(num) / np.sum(weights)
    return pred


def distance_calc_test(x_data):
    args = []
    for i in range(x_data.shape[2]):
        for j in range(i + 1, x_data.shape[2]):
            args.append((i, j))

    for num_threads in range(1, multiprocessing.cpu_count()):
        print("num_threads: {}".format(num_threads))
        now = time.time()
        with WorkerPool(n_jobs=num_threads, shared_objects=x_data) as pool:
            pool.map(get_distance, args)
        print("time: {}".format(time.time() - now))


if __name__ == "__main__":
    test_data = data_loader.get_test_data_sets(as_r=False)[0]
    x = test_data.get_x()
    y = test_data.get_y()

    description = test_data.get_descriptor()

    h = choose_h(x, y, description, False)
    y_hat = rbf_reg(get_x_distance_matrix(x, description), h, y)
    # noinspection PyTypeChecker
    print(stats.spearmanr(y, y_hat))
    print(h)
