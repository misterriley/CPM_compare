import re
import warnings

import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
import sklearn.metrics
from scipy.stats import ConstantInputWarning
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd

# MAT_FILES_PATH = "G:/.shortcut-targets-by-id/1Y42MQjJzdev5CtNSh2pJh51BAqOrZiVX/IMAGEN/CPM_mat/"
MAT_FILES_PATH = "G:/My Drive/CPM_test_data"
P_THRESH = 0.05
COLLAPSE_VECTORS = True

class CPM_masker:

    def __init__(self, x, y):
        self.corrs = [stats.pearsonr(a, y) for a in x.transpose()]
        self.x = x
        self.y = y

    def get_x(self, threshold, mask_type, x=None):
        if x is None:
            x = self.x
        pos_mask = [c[1] < threshold and c[0] > 0 for c in self.corrs]
        neg_mask = [c[1] < threshold and c[0] < 0 for c in self.corrs]

        pos_masked = x[:, pos_mask]
        neg_masked = x[:, neg_mask]

        if COLLAPSE_VECTORS:
            pos_masked = pos_masked.sum(axis=1).reshape(-1, 1)
            neg_masked = neg_masked.sum(axis=1).reshape(-1, 1)

        if mask_type == "positive":
            ret = pos_masked
        elif mask_type == "negative":
            ret = neg_masked
        else:
            if COLLAPSE_VECTORS:
                ret = pos_masked - neg_masked # if collapsing the vectors
            else:
                ret = x[:, [pos_mask[i] or neg_mask[i] for i in range(len(pos_mask))]]

        return ret

def run_lasso_comparison(x, y, cv=5, n_jobs=-1):

    x_normed = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    for i in np.logspace(-3.5, -1.5, num=101, base=10):
        model = Lasso(alpha=i)
        scores = cross_val_score(model, x_normed, y, cv=cv, n_jobs=n_jobs)
        print(f"alpha = {i:.4f}: {scores.mean():.4f}")
        model.fit(x_normed, y)
        non_zero_betas = np.count_nonzero(model.coef_)
        print(f"non_zero_betas = {non_zero_betas}")

def get_masker(x, y):
    return CPM_masker(x, y)

if __name__ == '__main__':

    import logging
    logging.captureWarnings(True)

    for file in os.listdir(MAT_FILES_PATH):

        if not file.endswith(".mat"):
            continue

        print(f"Loading {file}")
        mat_file = sio.loadmat(os.path.join(MAT_FILES_PATH, file))

        y = pd.read_csv(os.path.join(MAT_FILES_PATH, "txnegop.txt"), sep="\t", header=None)
        y = y.values.reshape(-1)

        x = mat_file["stp_all"]
        num_peeps = x.shape[2]
        num_nodes = x.shape[1]
        utix = np.triu_indices(num_nodes, k=1)
        x = x[utix[0], utix[1], :].transpose()

        RUN_COMPARISON = False
        if RUN_COMPARISON:
            run_lasso_comparison(x, y, cv=len(y))

        fold_x_arr = [np.concatenate((x[:i], x[i + 1:]), axis=0) for i in range(num_peeps)]
        fold_y_arr = [np.concatenate((y[:i], y[i + 1:]), axis=0) for i in range(num_peeps)]
        fold_masker_arr = [get_masker(fold_x_arr[i], fold_y_arr[i]) for i in range(num_peeps)]
        for desc in ["all", "positive", "negative"]:
            for alpha in np.logspace(np.log(.1), np.log(.00001), num=201, base=np.exp(1)):
                y_pred = []
                for i in range(num_peeps):
                    fold_masker = fold_masker_arr[i]
                    fold_x_masked = fold_masker.get_x(alpha, desc)
                    fold_y = fold_y_arr[i].reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(fold_x_masked, fold_y.reshape(-1, 1))
                    pred_x = fold_masker.get_x(alpha, desc, x[i].reshape(1, -1))
                    y_pred.append(model.predict(pred_x)[0])
                rho = stats.spearmanr(y, y_pred)[0]
                print(f"alpha = {alpha:.6f}: desc = {desc}: rho = {rho:.4f}")

        print("\n")
