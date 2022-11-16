import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn import datasets
from sys import exit

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost_func(y, ytilde):
    loss = -np.mean(y * (np.log(ytilde)) - (1 - y) * np.log(1 - ytilde))
    return los
def grad_cf(X, y, ytilde):
    m = X.shape[0]
    gradW = - 1 / m * X.T @ (ytilde - y)
    gradb = 1 / m * np.sum((ytilde - y))
    return gradW, gradb

def train(X, y, M, n_epochs, eta):
    m, n = X.shape
    b = 0
    w = np.zeros((n,1))
    errors = []
    n_iterations = n // M
    data_indices = np.arange(n)
    for epoch in range(n_epochs):
        for j in range(n_iterations):
            # pick datapoints with replacement
            chosen_datapoints = np.random.choice(
                data_indices, size=M, replace=False
            )

            # minibatch training data
            X_k = X[chosen_datapoints]
            y_k = y[chosen_datapoints]

            ytilde = sigmoid(X_k @ w + b)
            dw, db = grad_cf(X_k, y_k, ytilde)
            w -= eta * dw
            b -= eta * db
        errors.append(cost_func(y, sigmoid(X @ w + b)))

    return w, b, errors

if __name__ == "__main__":
    pass
