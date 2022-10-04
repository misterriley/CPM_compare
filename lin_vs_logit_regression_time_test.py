import timeit
import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor

size = 1000
count = 35000


def generate_data_set():
    y = np.random.randint(0, 2, size=size)
    x = np.random.random(size=(size, 1))
    return x, y


def poisson_regression_test():

    lr = PoissonRegressor()
    for i in range(count):
        x, y = generate_data_set()
        lr.fit(x, y)


def logistic_regression_test():
    lr = LogisticRegression()
    for i in range(count):
        x, y = generate_data_set()
        lr.fit(x, y)


def linear_regression_test():
    lr = LinearRegression()
    for i in range(count):
        x, y = generate_data_set()
        lr.fit(x, y)


def t_test_test():
    for i in range(count):
        x, y = generate_data_set()
        ttest_ind(x[y == 0], x[y == 1])


if __name__ == "__main__":

    print("Poisson regression: ", timeit.timeit(poisson_regression_test, number=1))
    print("Logistic regression: ", timeit.timeit(logistic_regression_test, number=1))
    print("Linear regression: ", timeit.timeit(linear_regression_test, number=1))
    print("T-test: ", timeit.timeit(t_test_test, number=1))