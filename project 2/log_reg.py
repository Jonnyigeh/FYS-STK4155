import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from autograd import grad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sys import exit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(0)
class log_regressor():
    """Class for logistic regression used for binary classification (hard classifier)
            The default values are chosen for classifying the WBC dataset.

    args:
        X_data                      (np.array): Input data
        Y_data                      (np.array): Target output data
        epochs                           (int): Number of epochs (default=50)
        batch_size                       (int): Size of minibatches (default=50)
        eta                            (float): Learning rate (default=0.3)
        lmbd                           (float): Regularization parameter (default=0.1)
    """
    def __init__(self,
            X_data,
            y_data,
            batch_size=50,
            epochs=50,
            eta=0.3,
            lmbda=0.1):

<<<<<<< HEAD
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
=======
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
        """Cost function for the logistic regressor.
            From log-likelihood (Cross entropy)

        args:
            ytilde          (np.array): Model data to be compared to target (self.y)

        returns:
            loss            (np.array): The loss of model vs target
        """
        loss = -np.mean(self.y * (np.log(ytilde)) - (1 - self.y) * np.log(1 - ytilde))

        return loss

    def grad_cf(self, X, y, ytilde):
        """Gradient of the cross entropy

        args:
            X               (np.array): Input data
            y               (np.array): target data
            ytilde          (np.array): model data

        returns:
            gradW           (np.array): Gradient of cf wrt. weights
            gradb           (np.array): Gradient of cf wrt. bias
        """
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
>>>>>>> 3e6dcaefc03588ca4d44169d318dcebda49f7b12

if __name__ == "__main__":
    # This code will produce the heat map shown in result section (Logistic Regressino: Classification)
    data, target = datasets.load_breast_cancer(return_X_y=True)
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)
    SS = StandardScaler()                        # Standard scaling the data to keep the network behaving nicely
    train_data = SS.fit_transform(train_data)
    test_data = SS.fit_transform(test_data)
    eta_vals = np.logspace(-6, 0, 7)
    lmbd_vals = np.logspace(-6, 0, 7)
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbd_vals):
            lr = log_regressor(train_data, train_target, eta=eta, lmbda=lmbda)
            lr.train()
            ytilde = lr.predict(test_data)
            yt_train = lr.predict(train_data)
            try:
                train_accuracy[i][j] = accuracy_score(train_target.reshape(yt_train.shape), yt_train)
                test_accuracy[i][j] = accuracy_score(test_target.reshape(ytilde.shape), ytilde)
            except ValueError:
                breakpoint()

    fig, axs = plt.subplots(2, figsize=(8,8))
    sns.heatmap(test_accuracy, annot=True, ax=axs[0], fmt=".2f",
                            xticklabels=eta_vals, yticklabels=lmbd_vals)
    sns.heatmap(train_accuracy, annot=True, ax=axs[1], fmt=".2f",
                            xticklabels=eta_vals, yticklabels=lmbd_vals)
    fig.suptitle("Logistic regression using SGD")
    axs[0].set_title("Test data")
    axs[0].set_xlabel(r"Regularizaton parameter: $\lambda$")
    axs[0].set_ylabel(r"Learning rate: $\eta$")
    axs[1].set_title("Train data")
    axs[1].set_xlabel(r"Regularizaton parameter: $\lambda$")
    axs[1].set_ylabel(r"Learning rate: $\eta$")
    fig.tight_layout()
    # plt.savefig("log_reg_50epoch_bs50_heatmap.pdf")
    # plt.show()
