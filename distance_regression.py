import time

import numpy as np
import torch
from mpire import WorkerPool
from scipy import stats

import data_loader
import utils

from sklearn.model_selection import KFold


def dist_between_matrices(a, b):
    # https: // gmarti.gitlab.io / math / 2019 / 12 / 25 / riemannian - mean - correlation.html
    _, eigenvals, _ = torch.linalg.svd(torch.linalg.solve(a, b))
    return torch.sqrt(torch.sum(torch.log(eigenvals) ** 2))


def get_distance(x, i, j):
    print(i, j)
    return i, j, dist_between_matrices(x[:, :, i], x[:, :, j])


def get_x_distance_matrix(x):
    now = time.time()
    print("Starting distance matrix calculation")
    with WorkerPool(n_jobs=1, shared_objects=x) as pool:
        args = []
        for i in range(x.shape[2]):
            for j in range(i + 1, x.shape[2]):
                args.append((i, j))
        results = pool.map(get_distance, args)
    print("time: {}".format(time.time() - now))

    dists = np.zeros((x.shape[2], x.shape[2]))
    for i, j, dist in results:
        dists[i, j] = dist
        dists[j, i] = dist
    return dists


def get_y_distance_matrix(y):
    dists = np.zeros((y.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(i + 1, y.shape[0]):
            dists[i, j] = abs(y[i] - y[j])
            dists[j, i] = dists[i, j]
    return dists


def fit(x, y):
    x_dist_matrix = get_x_distance_matrix(x)
    y_dist_matrix = get_y_distance_matrix(y)

    cov = 0
    var = 0
    for i in range(x.shape[2]):
        for j in range(i + 1, x.shape[2]):
            cov += x_dist_matrix[i, j] * y_dist_matrix[i, j]
            var += y_dist_matrix[i, j] ** 2
    k = cov / var
    return k, x_dist_matrix, y_dist_matrix


def fit_x_to_y(k, y, known_x, known_y):
    # assuming that x must be a correlation matrix
    n_entries = (known_x.shape[0] ** 2 - known_x.shape[0]) / 2
    x_vector = 2 * torch.rand(size=(int(n_entries), 1)) - 1
    x_vector.requires_grad = True

    optimizer = torch.optim.Adam([x_vector], lr=0.1)
    y = torch.tensor(y).float()
    known_x_list = known_x.detach().unbind(2)

    upper_indices = np.triu_indices(known_x.shape[0], 1)
    torch.autograd.set_detect_anomaly(True)

    min_loss = float("inf")
    best = None

    steps_since_improvement = 0
    abs_y_diff = torch.abs(y - known_y)
    for j in range(100000):
        optimizer.zero_grad()

        a = ut_to_full(known_x, upper_indices, x_vector)

        x_dists = torch.stack([dist_between_matrices(a, x) for x in known_x_list])
        loss_ = ((k * abs_y_diff - x_dists) ** 2).sum()
        loss_.backward()

        min_loss = min(loss_.item(), min_loss)
        if min_loss == loss_.item():
            steps_since_improvement = 0
            best = a.detach().numpy()
        else:
            steps_since_improvement += 1

        if steps_since_improvement > 100:
            break

        old_x = x_vector.detach().clone()

        optimizer.step()

        if x_vector.isnan().any():
            print("nan")
            break
        print(loss_.item(), steps_since_improvement)

    print(min_loss)
    return ut_to_full(known_x, upper_indices, best)


def ut_to_full(known_x, upper_indices, x_vector):
    a = torch.eye(known_x.shape[0]).float()
    transformed_x = torch.tanh(x_vector).squeeze()  # tanh to keep values between -1 and 1
    a[upper_indices] = transformed_x
    a.T[upper_indices] = transformed_x
    return a


def fit_y_to_x(k, x, known_x_list, known_y_list):

    y = torch.tensor(known_y_list.mean()).float().requires_grad_()
    optimizer = torch.optim.Adam([y], lr=0.1)
    x_dists = torch.stack([dist_between_matrices(a, x) for a in known_x_list.unbind(2)])
    known_y_list = torch.tensor(known_y_list)

    torch.autograd.set_detect_anomaly(True)

    min_loss = float("inf")
    best = None

    steps_since_improvement = 0
    for j in range(100000):
        optimizer.zero_grad()

        loss_ = ((k * torch.abs(y - known_y_list) - x_dists) ** 2).sum()
        loss_.backward()

        last_min = min_loss
        min_loss = min(loss_.item(), min_loss)
        if min_loss < last_min:
            steps_since_improvement = 0
            best = y.detach().item()
        else:
            steps_since_improvement += 1

        if steps_since_improvement > 100:
            break

        optimizer.step()
        print(loss_.item(), steps_since_improvement, y.item())

    print(min_loss)
    return y.detach().item()


def main():
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
        x = torch.from_numpy(data_set.x).float()
        y = data_set.y
        x_as_r = utils.fisher_z_to_r(x)
        for i in range(x.shape[0]):
            x_as_r[i, i, :] = 1  # the diagonal would be ~.99999 otherwise
        kf = KFold(n_splits=5, shuffle=True).split(y)
        y_hat = [None] * len(y)
        for train_index, test_index in kf:
            split_x_as_r = x_as_r[:, :, train_index]
            split_y = y[train_index]
            k, x_dist_matrix, y_dist_matrix = fit(split_x_as_r, split_y)
            for ti in test_index:
                y_hat[ti] = fit_y_to_x(k, x_as_r.unbind(2)[ti], split_x_as_r, split_y)

        print(stats.spearmanr(y, y_hat))


if __name__ == "__main__":
    main()
