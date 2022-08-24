import time

from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import data_loader
import numpy as np
from mpire import WorkerPool
import torch

PATIENCE = 200
NUM_REPETITIONS = 5
N_JOBS = 1
K_FOLDS = 10


def gradient_descent(y_, x_):
    n_vectors = 2

    proj_vectors = 2 * torch.rand(size=(x_.shape[1], n_vectors)) - 1
    proj_vectors = proj_vectors.div(torch.norm(proj_vectors, p=2,dim=0))
    reg_vector = 2 * torch.rand(size=(n_vectors + 1, 1)) - 1

    proj_vectors.requires_grad = True
    reg_vector.requires_grad = True
    optimizer = torch.optim.Adam([proj_vectors, reg_vector], lr=0.03)
    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).float()

    now = time.time()

    steps_without_improvement = 0
    best_loss = np.inf
    for i in range(100000):
        optimizer.zero_grad()
        proj_vars = (proj_vectors.T @ x_ @ proj_vectors).squeeze()
        output = reg_vector[2] * torch.log(proj_vars[:,0]) + reg_vector[1]*torch.log(proj_vars[:,1]) + reg_vector[0]
        loss = torch.mean(torch.abs(output - y_))

        if loss.item() < best_loss:
            best_loss = loss.item()
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if time.time() - now > 1:
            print("loss: ", loss.item())
            print("steps taken: {}".format(i))
            now = time.time()

        with torch.no_grad():
            norm = proj_vector.norm(p=2, dim=0, keepdim=True)
            proj_vector = proj_vector / norm.item()

        loss.backward()
        optimizer.step()

        if steps_without_improvement > PATIENCE:
            break

    return proj_vector.detach().numpy(), reg_vector.detach().numpy()


def predicted_output(proj_, x_, reg_vector_):
    return reg_vector_[1] * np.log((proj_.T @ x_ @ proj_).squeeze()) + reg_vector_[0]


def job(shared_objects, k_):
    kf_split_, x_, y_ = shared_objects

    train_, test_ = kf_split_[k_]
    train_x_ = x_[train_]
    test_x_ = x_[test_]
    train_y_ = y_[train_]
    test_y_ = y_[test_]

    proj_, reg = gradient_descent(train_y_, train_x_)

    test_predictions_ = predicted_output(proj_, test_x_, reg)
    return test_predictions_, test_y_


def torch_variance_estimation():
    sets = data_loader.get_test_data_sets(as_r=True, clean_data=True)
    for data_set in sets:
        x = data_set.x
        y = data_set.y

        x = np.transpose(x, (2, 0, 1))

        correlations = [None] * NUM_REPETITIONS
        for i in range(NUM_REPETITIONS):
            kf = KFold(n_splits=K_FOLDS, shuffle=True)
            kf_split = list(kf.split(y))

            with WorkerPool(n_jobs=N_JOBS, shared_objects=(kf_split, x, y)) as pool:
                results = pool.map(job, range(K_FOLDS))

            test_predictions = np.hstack(np.array([x for x, y in results], dtype=object)).astype(np.float32)
            test_y = np.hstack(np.array([y for x, y in results], dtype=object)).astype(np.float32)
            corr = spearmanr(test_predictions, test_y)
            print("Correlation: {}".format(corr))
            correlations[i] = corr[0]
        print("Mean correlation: {}".format(np.mean(correlations)))
        print("Std correlation: {}".format(np.std(correlations)))
        print("Max correlation: {}".format(np.max(correlations)))
        print("Min correlation: {}".format(np.min(correlations)))


if __name__ == "__main__":
    torch_variance_estimation()
