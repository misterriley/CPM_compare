import time

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import data_loader
import numpy as np
from mpire import WorkerPool
from multiprocessing import Lock
import torch
import seaborn as sns
import utils


def gradient_descent(y_, x_, ridge_c_):
    proj_vector = 2 * torch.rand(size=(x_.shape[0], 1)) - 1
    proj_vector /= torch.norm(proj_vector)
    reg_vector = 2 * torch.rand(size=(2, 1)) - 1

    proj_vector.requires_grad = True
    reg_vector.requires_grad = True
    optimizer = torch.optim.Adam([proj_vector, reg_vector], lr=0.000003)
    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).float()

    now = time.time()

    for i in range(100000):
        optimizer.zero_grad()
        proj_var = (proj_vector.T @ (proj_vector.T @ x_).squeeze()).squeeze()
        output = reg_vector[1] * torch.log(proj_var) + reg_vector[0]
        loss = torch.sum((output - y_) ** 2)
        loss_with_ridge = loss + ridge_c_ * torch.sum(proj_vector ** 2) + ridge_c_ * torch.sum(reg_vector ** 2)
        loss_with_ridge.backward()
        optimizer.step()

        if time.time() - now > 3:
            print("loss: ", loss_with_ridge.item())
            gn = proj_vector.grad.detach().data.norm() ** 2 + reg_vector.grad.detach().data.norm() ** 2
            print("gnorm:", np.sqrt(gn.item()))
            print("")
            now = time.time()

    return proj_vector.detach().numpy(), reg_vector.detach().numpy()


def predicted_output(proj_, x_, reg_vector_):
    return reg_vector_[1] * np.log(np.matmul(np.matmul(proj_, x_), proj_)) + reg_vector_[0]


def job(shared_objects, k_):
    kf_split_, x_, y_, lock, ridge_c_ = shared_objects

    train_, test_ = kf_split_[k_]
    train_x_ = x_[:, :, train_]
    test_x_ = x_[:, :, test_]
    train_y_ = y_[train_]
    test_y_ = y_[test_]

    proj_, reg = gradient_descent(train_y_, train_x_, ridge_c_)

    test_predictions_ = np.array([predicted_output(proj_, test_x_[:, :, t], reg) for t in range(len(test_))])
    return test_predictions_, test_y_


N_JOBS = 1  # multiprocessing.cpu_count() - 2
K_FOLDS = N_JOBS * 4


def torch_variance_estimation():
    dl = data_loader.DataLoader(
        protocol_c=[
            # "IMAGEN",
            # "sadie-marie",
            "test_data"
        ],
        file_c=None,
        # [
        # "mats_sst_fu2.mat",
        # "rest_estroop_acc_interf_2460_cpm_ready.mat",
        # "enback_estroop_acc_interf_2460_cpm_ready.mat",
        # "stp_all_clean2.mat"
        # ],
        y_col_c=None,
        clean_data=True)
    for data_set in dl.data_sets:
        x = data_set.x
        y = data_set.y
        x_as_r = utils.fisher_z_to_r(x)
        for i in range(x.shape[0]):
            x_as_r[i, i, :] = 1  # the diagonal would be ~.99999 otherwise

        ridge_test_df = pd.DataFrame(columns=["ridge_c", "mse", "mse_std", "spearman_r", "spearman_p"])

        for ridge_c in [0, *np.logspace(-6, 0, num=7, base=10)]:
            print("starting ridge_c {}".format(ridge_c))

            kf = KFold(n_splits=K_FOLDS, shuffle=True)
            kf_split = list(kf.split(y))

            with WorkerPool(n_jobs=N_JOBS, shared_objects=(kf_split, x_as_r, y, Lock(), ridge_c)) as pool:
                results = pool.map(job, range(K_FOLDS))

            test_predictions = np.hstack(np.array([x for x, y in results], dtype=object)).astype(np.float32)
            test_y = np.hstack(np.array([y for x, y in results], dtype=object)).astype(np.float32)
            corr = spearmanr(test_predictions, test_y)
            print("Correlation: {}".format(corr))

            ridge_test_df.loc[len(ridge_test_df)] = [ridge_c, mean_squared_error(test_y, test_predictions),
                                                     np.std(test_predictions - test_y), corr[0], corr[1]]
        ridge_test_df.to_csv("{}_ridge_test_df.csv".format(data_set.descriptor))


MIN_EIGEN_VALUE = 0.00001


def corr_matrix_reimann_dist(a, b):
    eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(a).dot(b))
    return np.sqrt(np.sum(np.log(eigenvals) ** 2))





def matrix_inversion_test():
    dl = data_loader.DataLoader(
        protocol_c=[
            # "IMAGEN",
            # "sadie-marie",
            "test_data"
        ],
        file_c=None,
        # [
        # "mats_sst_fu2.mat",
        # "rest_estroop_acc_interf_2460_cpm_ready.mat",
        # "enback_estroop_acc_interf_2460_cpm_ready.mat",
        # "stp_all_clean2.mat"
        # ],
        y_col_c=None,
        clean_data=True)
    for data_set in dl.data_sets:
        x = data_set.x
        y = data_set.y
        x_as_r = utils.fisher_z_to_r(x)
        for i in range(x.shape[0]):
            x_as_r[i, i, :] = 1  # the diagonal would be ~.99999 otherwise

        #exhaustive_dist_search(x_as_r)
        sample_dist_search(x_as_r, y)


def sample_dist_search(x, y):

    scatter_x = []
    scatter_y = []

    for i in range(5000):
        ids = np.random.choice(x.shape[2], 2, replace=False)
        a = x[:, :, ids[0]]
        b = x[:, :, ids[1]]
        dist = utils.wasserstein_distance_cov(a, b)
        y_diff = abs(y[ids[0]] - y[ids[1]])
        scatter_x.append(dist)
        scatter_y.append(y_diff)

    print(spearmanr(scatter_x, scatter_y))
    plt.scatter(scatter_x, scatter_y)
    z = np.polyfit(scatter_x, scatter_y, 1)
    p = np.poly1d(z)
    plt.plot(scatter_x, p(scatter_x), "r--")
    plt.show(block=True)


def exhaustive_dist_search(x):
    dists = np.array([])
    for i in range(x.shape[2]):
        a = x[:, :, i]
        print(i)
        for j in range(i + 1, x.shape[2]):
            b = x[:, :, j]
            now = time.time()

            # u, s, w = np.linalg.svd(np.linalg.solve(a, b))  # use svd instead of eig to avoid numerical issues,
            # similarly use solve instead of inv to avoid numerical issues

            # thanks to https://gmarti.gitlab.io/math/2019/12/25/riemannian-mean-correlation.html
            dist = utils.wasserstein_distance_cov(a, b)
            dists = np.append(dists, dist)
            print("dist: {} --- time: {}".format(dist, time.time() - now))
    sns.displot(dists)
    plt.show(block=True)


if __name__ == "__main__":
    # torch_variance_estimation()
    matrix_inversion_test()
