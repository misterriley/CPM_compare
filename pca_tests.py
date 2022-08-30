import time

import numpy as np
import torch
from mpire import WorkerPool
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import data_loader

PATIENCE = 3000
NUM_REPETITIONS = 20
N_JOBS = 1
K_FOLDS = 10
N_VECTORS = 1
LEARNING_RATE = 0.1


def gradient_descent(y_, x_, l1_constant):
    proj_vectors = 2 * torch.rand(size=(x_.shape[1], N_VECTORS)) - 1
    reg_vector = 2 * torch.rand(size=(N_VECTORS + 1, 1)) - 1
    reg_vector[0] = y_.mean()

    proj_vectors.requires_grad = True
    reg_vector.requires_grad = True
    optimizer = torch.optim.Adam([proj_vectors, reg_vector], lr=LEARNING_RATE)
    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).float()

    now = time.time()

    steps_without_improvement = 0
    best_loss = np.inf
    best_proj_vectors = None
    best_reg_vector = None
    best_l1_norm = np.inf
    i = 0
    last_lips = 0
    lips_window = PATIENCE
    for i in range(100000):
        with torch.no_grad():
            for pv_index_1 in range(N_VECTORS):
                v = proj_vectors[:, pv_index_1]
                norm = v.norm(p=2, dim=0, keepdim=True)
                proj_vectors[:, pv_index_1] = v / norm.item()
                for pv_index_2 in range(pv_index_1 + 1, N_VECTORS):
                    u = proj_vectors[:, pv_index_2]
                    proj_u_on_v = torch.dot(u, v) / torch.dot(v, v) * v
                    proj_vectors[:, pv_index_2] = u - proj_u_on_v

        optimizer.zero_grad()
        output = predicted_output(proj_vectors, x_, reg_vector)
        l1_norm = torch.sum(torch.abs(proj_vectors))
        loss = torch.sum((output - y_) ** 2) + l1_constant * l1_norm

        if loss.item() < best_loss:
            steps_since_improvement = steps_without_improvement + 1
            current_lips = (best_loss - loss.item()) / steps_since_improvement if best_loss is not np.inf else 0
            lips = current_lips * min(steps_since_improvement, lips_window) + \
                   last_lips * max(0, lips_window - steps_since_improvement)
            lips /= lips_window
            last_lips = lips

            best_loss = loss.item()
            best_proj_vectors = proj_vectors.detach().numpy()
            best_reg_vector = reg_vector.detach().numpy()
            best_l1_norm = l1_norm.item()

            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            lips = last_lips * max(0, lips_window - steps_without_improvement) / lips_window

        if time.time() - now > 1:
            print("")
            print("loss: {:6.2f} (l1: {:6.2f}, ssq: {:6.2f})".format(best_loss, best_l1_norm * l1_constant,
                                                                     best_loss - best_l1_norm * l1_constant))
            print("steps taken: {}".format(i))
            print("l1 norm: {:6.2f}".format(best_l1_norm))
            print("small components: {}".format(np.count_nonzero(np.abs(best_proj_vectors) < 1e-4)))
            print("steps without improvement: {}".format(steps_without_improvement))
            print("lips: {:6.4f} ({:6.2f}/{})".format(lips, lips * lips_window, lips_window))
            now = time.time()

        loss.backward()
        optimizer.step()

        if steps_without_improvement > PATIENCE:
            break

    print("final loss: ", best_loss)
    print("total steps taken: {}".format(i))
    print("nonzero proj_vectors: {}".format(np.count_nonzero(best_proj_vectors)))
    return best_proj_vectors, best_reg_vector


def predicted_output(proj_, x_, reg_vector_):
    is_torch = isinstance(x_, torch.Tensor)
    if not is_torch:
        proj_ = torch.from_numpy(proj_).float()
        x_ = torch.from_numpy(x_).float()
        reg_vector_ = torch.from_numpy(reg_vector_).float()

    proj_vars = torch.sum((proj_.T @ x_) * proj_.T, dim=-1)
    output = (reg_vector_[0] + reg_vector_[1:].T @ proj_vars.T).squeeze()
    return output if is_torch else output.detach().numpy()


def job(shared_objects, k_):
    print("Starting fold {}".format(k_))
    kf_split_, x_, y_, l1_constant = shared_objects

    train_, test_ = kf_split_[k_]
    train_x_ = x_[train_]
    test_x_ = x_[test_]
    train_y_ = y_[train_]
    test_y_ = y_[test_]

    proj_, reg = gradient_descent(train_y_, train_x_, l1_constant)

    test_predictions_ = predicted_output(proj_, test_x_, reg)
    return test_predictions_, test_y_


# noinspection PyTypeChecker
def torch_variance_estimation():
    sets = data_loader.get_test_data_sets(as_r=True, clean_data=True)
    for data_set in sets:
        x = data_set.x
        y = data_set.y

        x = np.transpose(x, (2, 0, 1))

        correlations = [None] * NUM_REPETITIONS
        for i in range(NUM_REPETITIONS):
            print("Repetition {}".format(i))
            kf = KFold(n_splits=K_FOLDS, shuffle=True)
            kf_split = list(kf.split(y))
            l1_constant = 1000

            with WorkerPool(n_jobs=N_JOBS, shared_objects=(kf_split, x, y, l1_constant)) as pool:
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
