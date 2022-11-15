import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from autograd import grad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn import datasets
from sys import exit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(0)
class log_regressor():
    def __init__(self,
            X_data,
            y_data,
            batch_size=10,
            epochs=50,
            eta=0.3,
            lmbda=0.1):

        self.n, self.n_features = X_data.shape
        self.b = 0
        self.X = X_data
        self.y = y_data
        self.eta = eta
        self.w = np.zeros((self.n_features,1))
        self.errors = []
        self.M = batch_size
        self.n_epochs = epochs
        self.n_iterations = self.n // self.M
        self.lmbda = lmbda

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cost_func(self, ytilde):
        loss = -np.mean(self.y * (np.log(ytilde)) - (1 - self.y) * np.log(1 - ytilde))

        return loss

    def grad_cf(self, X, y, ytilde):
        m = X.shape[0]

        gradW = - 1 / m * X.T @ (y.reshape(ytilde.shape) - ytilde)
        gradb = 1 / m * np.sum((y.reshape(ytilde.shape) - ytilde))

        return gradW, gradb

    def predict(self, X):
        probs = self.sigmoid(X @ self.w + self.b)
        probs[probs<0.5] = 0
        probs[probs>=0.5] = 1
        prediction = probs
        return prediction


    def train(self):
        """Trains the logistic regressor weights
                using Stochastic gradient descent (Ridge or OLS depending on lmbda value)

        """
        data_indices = np.arange(self.n)
        for epoch in range(self.n_epochs):
            for j in range(self.n_iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.M, replace=False
                )

                # minibatch training data
                X_k = self.X[chosen_datapoints]
                y_k = self.y[chosen_datapoints]

                probability = self.sigmoid(X_k @ self.w + self.b)
                dw, db = self.grad_cf(X_k, y_k, probability)

                if self.lmbda > 0.0:
                    dw += self.lmbda * self.w
                    db += self.lmbda * self.b

                self.w -= self.eta * dw
                self.b -= self.eta * db

            ytilde = self.sigmoid(self.X @ self.w + self.b)
            self.errors.append(self.cost_func(self.sigmoid(self.X @ self.w + self.b)))

if __name__ == "__main__":
    data, target = datasets.load_breast_cancer(return_X_y=True)
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

    eta_vals = np.logspace(-3, 0, 10)
    lmbd_vals = np.logspace(-3, 0, 10)
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbd_vals):
            lr = log_regressor(train_data, train_target, eta=eta, lmbda=lmbda)
            lr.train()
            ytilde = lr.predict(test_data)
            yt_train = lr.predict(train_data)
            try:
                train_accuracy[i][j] = accuracy_score(yt_train, train_target.reshape(yt_train.shape))
                test_accuracy[i][j] = accuracy_score(ytilde, test_target.reshape(ytilde.shape))
            except ValueError:
                breakpoint()
        # print("Accuracy score on training data " + str(accuracy_score(yt_train, train_target.reshape(yt_train.shape))))
        # print("Accuracy score on test data: " + str(accuracy_score(ytilde, test_target.reshape(ytilde.shape))))
        # print("\n")
    fig, axs = plt.subplots(2, figsize=(10,10))
    sns.heatmap(test_accuracy, annot=True, ax=axs[0], cmap="viridis")
    sns.heatmap(train_accuracy, annot=True, ax=axs[1], cmap="viridis")
    axs[0].set_title("Test")
    axs[1].set_title("Train")
    plt.show()
    breakpoint()
