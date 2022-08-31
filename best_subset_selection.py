import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import data_loader
from main import convert_to_wide
from abess import LinearRegression

from sort_and_regress import N_SPLITS


def aic_c(model, x_masked, y):
    aic = model.ic_
    k = model.support_size
    n = x_masked.shape[0]
    ret = aic + 2 * k * (k + 1) / (n - k - 1)
    return aic# + 2 * k * (k + 1) / (n - k - 1)


def main():
    ds = data_loader.get_imagen_data_sets(file_c="mats_mid_bsl.mat",
                                          #y_col_c="csi_c_sum_fu2",
                                          as_r=False,
                                          clean_data=True)
    for d in ds:
        x = d.x
        y = d.y

        x = convert_to_wide(x)
        const_index = (np.min(x, axis=0) == np.max(x, axis=0))

        x_masked = x[:, ~const_index]
        lengths = np.logspace(0, np.log10(500), num=40)
        lengths = np.unique(np.round(lengths))
        ic = np.zeros(lengths.shape[0])
        for i in range(lengths.shape[0]):
            s = int(lengths[i])
            print("starting with ", s)
            model = LinearRegression(support_size=s, ic_type="aic")
            model.fit(x_masked, y)
            ic[i] = aic_c(model, x_masked, y)

        plt.scatter(x=lengths, y=ic)
        plt.xlabel('support_size')
        plt.ylabel(model.ic_type)
        plt.title('Model selection via IC')
        plt.show(block=True)

        n_entries = lengths[np.argmin(ic)]
        print(cross_validated_regression(x_masked, y, n_entries), n_entries)


def cross_validated_regression(x, y, n_entries):
    print("starting cross-validated regression with ", n_entries)

    if n_entries == 0:
        return 0#np.repeat(y.mean(), y.shape[0])

    ret = np.ndarray(y.shape[0])
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression(support_size=n_entries)
        model.fit(x_train, y_train)
        ret[test_index] = model.predict(x_test)

    return spearmanr(ret, y)

def cut_code(ind, model, x_masked, y, d):
    if len(ind) > 0:
        print("estimated non-zero: ", ind)
        print("estimated coef: ", model.coef_[ind])
        # got a hit, let's do some cross-validation
        kf = KFold(n_splits=N_SPLITS, shuffle=True)
        y_hat = np.ndarray(shape=(y.shape[0]), dtype=object)
        for i, (train_index, test_index) in enumerate(kf.split(x_masked)):
            x_train, x_test = x_masked[train_index], x_masked[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(x_train, y_train)
            y_hat[test_index] = model.predict(x_test)
        print(spearmanr(y_hat, y))
    print("finished with ", d.get_descriptor())

if __name__ == '__main__':
    # import warnings

    # warnings.filterwarnings("ignore")
    main()
