from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import data_loader
import numpy as np
import seaborn as sns
from mpire import WorkerPool
import time


class MuZero:
    def __init__(self, n_components, n_features, n_classes, n_workers=1):
        self.n_components = n_components
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_workers = n_workers
        self.worker_pool = WorkerPool(n_workers)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.matrices = self.worker_pool.map(get_eig, [self.X for _ in range(self.n_components)])
        self.matrices = np.array(self.matrices)
        self.matrices = self.matrices[:, :, 1]
        self.matrices = self.matrices.reshape(self.n_components, self.n_features, self.n_features)
        self.matrices

    def transform(self, X):
        return self.worker_pool.map(self.transform_data, [X for _ in range(self.n_components)])

    def transform_data(self, X):
        return np.dot(X, self.matrices)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, proj_):
        self.m = np.zeros(proj_.shape)
        self.v = np.zeros(proj_.shape)

    def update(self, grad, proj_):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** (self.t + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.t + 1))
        proj_ -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.t += 1
        return proj_


class PCATests:

    def __init__(self):
        self.data_loader = data_loader.DataLoader(protocol_c=["IMAGEN"],
                                                  file_c=["mats_mid_bsl.mat"],
                                                  y_col_c=["kirby_c_estimated_k_all_trials_fu2"],
                                                  clean_data=True)


def fisher_r_to_z(r):
    return np.arctanh(r)


def fisher_z_to_r(z):
    return np.tanh(z)


def get_eig(shared_objects, i):
    x = shared_objects
    # print("Getting eig {}".format(i))
    return np.linalg.eig(x[:, :, i])


def loss(proj_, y_, matrices):
    return np.sum([(np.log(proj_.T @ matrices[:, :, j] @ proj_) - y_[j]) ** 2 for j in range(len(y_))])


def one_grad(proj_, y_, matrix):
    return 2 * (np.log(proj_.T @ matrix @ proj_) - y_) * 1 / (proj_.T @ matrix @ proj_) * 2 * (proj_.T @ matrix)


# gradient of the loss function with respect to proj
def gradient(proj_, y_, matrices):
    grads = [one_grad(proj_, y_[j], matrices[:, :, j]) for j in range(len(y_))]
    ret = np.sum(grads, axis=0)

    return ret


def gradient_descent(proj_, y_, matrices, tolerance=0.01, max_iter=1000):
    adam_optimizer = AdamOptimizer()
    adam_optimizer.initialize(proj_)

    for step in range(max_iter):
        grad = gradient(proj_, y_, matrices)
        proj_ = adam_optimizer.update(grad, proj_)
        if step % 100 == 0:
            print("proj norm: {}".format(np.linalg.norm(proj_)))
            print("loss: {}".format(loss(proj_, y_, matrices)))
            print("grad norm: {}".format(np.linalg.norm(grad)))

        if np.linalg.norm(grad) < tolerance * np.sqrt(y_.shape[0]):
            break
    return proj_


INITIAL_PROJ_LENGTH = 1
K_FOLDS = 10

if __name__ == "__main__":

    pca_tests = PCATests()
    for data_set in pca_tests.data_loader.data_sets:
        x = data_set.x
        y = data_set.y
        x_as_r = fisher_z_to_r(x)
        for i in range(x.shape[0]):
            x_as_r[i, i, :] = 1

        kf = KFold(n_splits=K_FOLDS, shuffle=True)
        for k in range(K_FOLDS):
            train, test = next(kf.split(x_as_r))
            train_x = x_as_r[:, :, train]
            test_x = x_as_r[:, :, test]
            train_y = y[train]
            test_y = y[test]

            initial_random_proj = 2 * np.random.rand(x.shape[0]) - 1
            initial_random_proj /= (np.linalg.norm(initial_random_proj) / INITIAL_PROJ_LENGTH)

            proj = gradient_descent(initial_random_proj, train_y, train_x)

            print("Projection: {}".format(proj))

            test_predictions = [np.log(proj.T @ x_as_r[:, :, t] @ proj) for t in test]
            spearman_rho = np.corrcoef(test_predictions, test_y)[0, 1]
            print("Spearman rho: {}".format(spearman_rho))
