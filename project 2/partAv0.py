import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# from random import random, seed'

class GD():
    """
    Class for gradient descent methods

    """
    def __init__(self, x, deg):
        self.x = x
        self.n = len(self.x)
        self.y = self.f(x) + 0.5 * np.random.randn(self.n)
        self.deg = deg
        self.beta0 = np.random.randn(self.deg+1, 1)
        self.DM = np.ones((self.n,self.deg+1))
        for i in range(1, self.deg+1):
            self.DM[:,i] = self.x ** i
        # self.DM = self.create_X(self.x,self.y,self.deg)

    def f(self, x, a0 = 2, a1 = 4, a2 = -1):
        """Simple one dimensional function of a polynomial
                of degree 2.

        args:
            x               (float/np.array): Value(s) to evaluate polynomial onto
            a0, a1, a2      (float): Parameters in the polynomial

        returns:
            poly             (float/np.array): The evaluated polymomial
        """
        poly = a0 + a1 * x + a2 * x **2

        return poly

    def create_X(self, x,y,n):
        """Generates the design matrix given a grid of
                x, y datapoints - for a polynomial of degree n.

        Args:
            x = grid of x-datapoints
            y = grid of y-datapoints
            n = degree of desired polynomial fit
        Returns
            X = Design matrix
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((n + 1) * (n + 2) / 2)
        X = np.ones((N,l))

        for i in range(1, n+1):
            q = int((i) * (i+1) / 2)
            for k in range(i+1):
                X[:,q+k] = (x **(i - k)) * (y **k)

        return X

    def MSE(self, Z, Z_model):
        """Calculates the mean squared error

        args:
            Z           (np.array): Measured data
            Z_model     (np.array): Model data

        returns:
            mse         (float)   : Mean squared error

        """
        mse = np.mean( (Z.ravel() - Z_model) **2)

        return mse

    def R2(self, Z, Z_model):
        """Calculates the R2 score

        args:
            Z           (np.array): Measured data
            Z_model     (np.array): Model data

        returns:
            r2         (float)   : R2 score

        """
        n = len(Z_model)
        Z_mean = np.mean(Z.ravel())
        r2 = 1 - np.sum((Z.ravel() - Z_model) ** 2) / np.sum((Z.ravel() - Z_mean) **2)
        return r2

    def PlainGD(self, eta = 0.01, eps = 10 ** (-8), lmbda = 0.1, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):
        """Performs linear regression using Plain Gradient Descent

        args:
            eta               (float): Learning rate
            eps               (float): Stopping criteria (when gradient is smaller than this value we stop)
            lmbda             (float): Ridge parameter lambda
            method            (str)  : Method of choice (either OLS or Ridge)
            check_equality    (Bool) : True or False -> if you want to compare with matrix inversion regression

        returns:
            beta           (np.array):   Array of beta parameters in linear model
        """
        n = self.n
        beta0 = self.beta0
        DM = self.DM
        y = self.y.reshape(n,1)
        if method=="OLS":
            with np.printoptions(precision=3):
                print(f"Performing OLS regression using PlainGD with initial values = " ,(*beta0))
            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                m_prev = 0
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                m = avg_time1 * m_prev + (1 - avg_time1) * g
                s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                m = m / (1 - avg_time1)
                s = s / (1 - avg_time2)
                i = 1
                beta = beta0 - eta * m / (np.sqrt(s) + delta)
                mse = []
                j = 0
                while np.linalg.norm(g) > eps and j < 1000:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    j += 1
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    mhat = m / (1 - avg_time1 ** i)
                    shat = s / (1 - avg_time2 ** i)

                    beta -= eta * mhat / (np.sqrt(shat) + delta)
                    m_prev = m
                    s_prev = s
            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)))
                delta = 1e-8
                G = gradC[0]@gradC[0].T
                dim1, dim2 = np.shape(G)
                I = np.eye(dim1, dim2)

                beta = beta0 - (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta0.shape) * gradC[0]
                mse = []
                i = 0
                while np.linalg.norm(gradC[-1]) > eps and i < 1000:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    gradC.append(2 / n * (DM.T @ (DM @ beta - y)))
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)

                    beta -= (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]
            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                s_prev = s
                beta = beta0 - eta * g / np.sqrt(s + delta)
                mse = []
                i = 0
                while np.linalg.norm(g) > eps and i < 1000:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    s = avg_time * s_prev + (1 - avg_time) * g ** 2
                    beta -= eta * g / np.sqrt(s + delta)
                    s_prev = s
            else:

                gradC = 2 / n * (DM.T @ (DM @ beta0 - y))
                beta = beta0 - eta * gradC
                mse = []
                i = 0
                # while i > 100:
                while np.linalg.norm(gradC) > eps and i < 1000:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    gradC = 2 / n * (DM.T @ (DM @ beta - y))
                    beta -= eta * gradC



            # ref_beta = np.linalg.pinv(DM.T @ DM) @ DM.T @ y

        if method=="Ridge":
            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                m_prev = 0
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta0
                m = avg_time1 * m_prev + (1 - avg_time1) * g
                s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                m = m / (1 - avg_time1)
                s = s / (1 - avg_time2)
                i = 1
                beta = beta0 - eta * m / (np.sqrt(s) + delta)
                mse = []
                j = 0

                while np.linalg.norm(g) > eps and j < 1000:
                    j += 1
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    mhat = m / (1 - avg_time1 ** i)
                    shat = s / (1 - avg_time2 ** i)

                    beta -= eta * mhat / (np.sqrt(shat) + delta)
                    m_prev = m
                    s_prev = s

            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta0)
                delta = 1e-8
                G = gradC[0]@gradC[0].T
                dim1, dim2 = np.shape(G)
                I = np.eye(dim1, dim2)
                beta = beta0 - (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta0.shape) * gradC[0]
                mse = []
                i = 0
                while np.linalg.norm(gradC[-1]) > eps and i < 1000:
                    i += 1
                    mse.append(self.MSE(self.y,self.DM@beta))
                    gradC.append(2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta)
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    beta -= (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]

            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                beta = beta0 - eta * g / np.sqrt(s + delta)
                mse = []
                i = 0
                while np.linalg.norm(g) > eps and i < 100:
                    i += 1
                    mse.append(self.MSE(self.y,self.DM@beta))
                    g = 2 / n * (DM.T @ (DM @ beta0 - y))
                    s = avg_time * s_prev + (1 - avg_time) * g ** 2
                    beta -= eta * g / np.sqrt(s + delta)
                    s_prev = s

            else:
                XT_X = DM.T @ DM
                dim1, dim2 = np.shape(XT_X)
                I = np.eye(dim1, dim2)
                gradC = 2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta0
                beta = beta0 - eta * gradC

                with np.printoptions(precision=3):
                    print(f"Performing Ridge regression using PlainGD with initial values = " ,(*beta0))
                mse = []
                i = 0
                while i < 1000:
                #while np.linalg.norm(gradC) > eps and i < 1000:
                    i += 1
                    mse.append(self.MSE(self.y,self.DM@beta))
                    gradC = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    beta -= eta * gradC

            # ref_beta = np.linalg.pinv(DM.T @ DM + lmbda * I ) @ DM.T @ y

        if check_equality:
            np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)

        return beta,mse

    def MGD(self, eta = 0.01, eps = 10 ** (-8), gamma = 0.9, lmbda = 0.1, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):

        """Performs linear regression using Plain Gradient Descent with momentum

        args:
            eta               (float): Learning rate
            eps               (float): Stopping criteria (when gradient is smaller than this value we stop)
            gamma             (float): Momentum parameter
            lmbda             (float): Ridge parameter lambda
            method            (str)  : Regreesion method of choice (either OLS or Ridge)
            check_equality    (Bool) : True or False -> if you want to compare with matrix inversion regression

        returns:
            beta            (np.array):   Array of beta parameters in linear model
        """
        n = self.n
        beta0 = self.beta0
        DM = self.DM
        y = self.y.reshape(n,1)
        v_prev = 0

        if method=="OLS":
            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                v_prev = 0
                m_prev = 0
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                m = avg_time1 * m_prev + (1 - avg_time1) * g
                s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                m = m / (1 - avg_time1)
                s = s / (1 - avg_time2)
                i = 1
                beta = beta0 - eta * m / (np.sqrt(s) + delta)
                mse = []
                j = 0
                while np.linalg.norm(g) > eps and j < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    j += 1
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    mhat = m / (1 - avg_time1 ** i)
                    shat = s / (1 - avg_time2 ** i)
                    v = v_prev * gamma + eta * m / (np.sqrt(s) + delta)
                    beta -= v
                    v_prev = v
                    m_prev = m
                    s_prev = s


            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)))
                delta = 1e-8
                v_prev = 0
                G = gradC[0]@gradC[0].T
                dim1, dim2 = np.shape(G)
                I = np.eye(dim1, dim2)
                beta = beta0 - (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta0.shape) * gradC[0]
                mse = []
                i = 0
                while np.linalg.norm(gradC[-1]) > eps and i < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    gradC.append(2 / n * (DM.T @ (DM @ beta - y)))
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    v = gamma * v_prev + (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]
                    beta -= v
                    v_prev = v


            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                s_prev = s
                beta = beta0 - eta * g / np.sqrt(s + delta)
                mse = []
                i = 0
                while np.linalg.norm(g) > eps and i < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    s = avg_time * s_prev + (1 - avg_time) * g ** 2
                    v = gamma * v_prev + eta * g / np.sqrt(s + delta)
                    beta -= v
                    s_prev = s
                    v_prev = v

            else:
                with np.printoptions(precision=3):
                    print(f"Performing OLS regression using MGD with initial values = " ,(*beta0))
                gradC = 2 / n * (DM.T @ (DM @ beta0 - y))
                v_prev = 0
                beta = beta0 - eta * gradC
                mse = []
                i = 0
                while np.linalg.norm(gradC) > eps and i < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    gradC = 2 / n * (DM.T @ (DM @ beta - y))
                    v = v_prev * gamma + eta * gradC
                    beta -= v
                    v_prev = v
                print(f"MSE momentum: {mse}")

            ref_beta = np.linalg.pinv(DM.T @ DM) @ DM.T @ y


        if method=="Ridge":
            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                v_prev = 0
                m_prev = 0
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta0
                m = avg_time1 * m_prev + (1 - avg_time1) * g
                s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                m = m / (1 - avg_time1)
                s = s / (1 - avg_time2)
                i = 1
                beta = beta0 - eta * m / (np.sqrt(s) + delta)
                mse = []
                j = 0
                while np.linalg.norm(g) > eps and j < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    mhat = m / (1 - avg_time1 ** i)
                    shat = s / (1 - avg_time2 ** i)
                    v = gamma * v_prev + eta * mhat / (np.sqrt(shat) + delta)
                    beta -= v
                    m_prev = m
                    s_prev = s
                    v_prev = v

            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta0)
                delta = 1e-8
                G = gradC[0]@gradC[0].T
                v_prev = 0
                dim1, dim2 = np.shape(G)
                I = np.eye(dim1, dim2)
                beta = beta0 - (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta0.shape) * gradC[0]
                mse = []
                i = 0
                while np.linalg.norm(gradC[-1]) > eps and i < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta)
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    v = gamma * v_prev + (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]
                    beta -= v
                    v_prev = v

            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                v_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                beta = beta0 - eta * g / np.sqrt(s + delta)
                mse = []
                i = 0
                while np.linalg.norm(g) > eps and i < 100:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    s = avg_time * s_prev + (1 - avg_time) * g ** 2
                    v = gamma * v_prev + eta * g / np.sqrt(s + delta)
                    beta -= v
                    s_prev = s
                    v_prev = v


            else:
                with np.printoptions(precision=3):
                    print(f"Performing Ridge regression using MGD with initial values = " ,(*beta0))
                XT_X = DM.T @ DM
                dim1, dim2 = np.shape(XT_X)
                I = np.eye(dim1, dim2)
                gradC = 2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta0
                beta = beta0 - eta * gradC
                v_prev = 0
                mse = []
                i = 0
                while i < 1000:
                #while np.linalg.norm(gradC) > eps and i < 1000:
                    mse.append(self.MSE(self.y,self.DM@beta))
                    i += 1
                    gradC = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    v = v_prev * gamma + eta * gradC
                    beta -= v
                    v_prev = v




            # ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y

        if check_equality:
            np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)


        return beta,mse

    def SGD(self,eta =0.01, M = 10, n_epochs = 100, lmbda = 0.1, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):

        """Performs linear regression using Stochastic Gradient Descent
                    using tunable learning rate. (see function *learning rate*)

        args:
            eta               (float): Learning rate
            M                 (int)  : Size of minibatches
            n_epochs          (int)  : Number of epochs (iterations of all minibatches)
            lmbda             (float): Ridge parameter lambda
            method            (str)  : Regreesion method of choice (either OLS or Ridge)
            check_equality    (Bool) : True or False -> if you want to compare with matrix inversion regression

        returns:
            beta           (np.array):   Array of beta parameters in linear model
        """
        t0, t1 = 5, 50
        def learning_rate(t):
            return t0 / (t1 + t)
        n = self.n
        beta0 = self.beta0
        DM = self.DM
        y = self.y.reshape(n,1)
        m_ = int(n / M)                                  # Number of minibatches
        beta = np.zeros_like(beta0)
        for i, element in enumerate(beta0):
            beta[i] = element

        if method=="OLS":
            with np.printoptions(precision=3):
                print(f"Performing OLS regression using SGD with initial values = " ,(*beta0))
            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                m_prev = 0
                s_prev = 0
                t = 0
                mse = []
                j = 0

                for epoch in range(n_epochs):
                    for i in range(m_):
                        t += 1
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        mhat = m / (1 - avg_time1 ** t)
                        shat = s / (1 - avg_time2 ** t)

                        beta -= eta * mhat / (np.sqrt(shat) + delta)
                        m_prev = m
                        s_prev = s


            elif ADAGRAD:
                gradC = []
                delta = 1e-8
                mse= []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j+=1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC.append(2 / M * (DMi.T @ (DMi @ beta - yi)))
                        k = len(gradC)
                        G = sum(gradC[j]@gradC[j].T for j in range(k))
                        dim1, dim2 = np.shape(G)
                        I = np.eye(dim1, dim2)
                        beta -= (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]


            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j+=1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        s = avg_time * s_prev + (1 - avg_time) * g ** 2
                        beta -= eta * g / np.sqrt(s + delta)
                        s_prev = s

            else:
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j+=1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        beta -= eta * gradC

            if check_equality:
                ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y
                np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)

        if method=="Ridge":
            with np.printoptions(precision=3):
                print(f"Performing Ridge regression using SGD with initial values = " ,(*beta0))

            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                m_prev = 0
                s_prev = 0
                t = 0
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        t += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        mhat = m / (1 - avg_time1 ** t)
                        shat = s / (1 - avg_time2 ** t)

                        beta -= eta * mhat / (np.sqrt(shat) + delta)
                        m_prev = m
                        s_prev = s

            elif ADAGRAD:
                gradC = []
                delta = 1e-8
                mse= []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC.append(2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta)
                        k = len(gradC)
                        G = sum(gradC[j]@gradC[j].T for j in range(k))
                        dim1, dim2 = np.shape(G)
                        I = np.eye(dim1, dim2)
                        beta -= (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]
                        mse.append(self.MSE(self.y,self.DM@beta))

            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        s = avg_time * s_prev + (1 - avg_time) * g ** 2
                        beta -= eta * g / np.sqrt(s + delta)
                        s_prev = s
                        mse.append(self.MSE(self.y,self.DM@beta))
            else:
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)              # Remove the upper m, since we add k*M + M meaning it'd crash
                                                                # if we by chance choose k = max(m)
                        eta = learning_rate(epoch * m_ + i)
                        DMi = DM[k*M:k*M + M]
                        yi = y[k*M:k*M + M]
                        XT_X = DMi.T @ DMi
                        dim1, dim2 = np.shape(XT_X)
                        I = np.eye(dim1, dim2)
                        gradC = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        beta -= eta * gradC



            if check_equality:
                # XT_X = DM.T @ DM
                # ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y
                np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)

        return beta,mse

    def SMGD(self,eta=0.01, M = 10, n_epochs = 100, gamma=0.9, lmbda = 0.1, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):
        """Performs linear regression using Stochastic Gradient Descent with momentum
                and a tunable learning rate. (see function learning rate).

        args:

            eta               (float): Learning rate
            gamma             (float): Momentum parameter
            M                 (int)  : Size of minibatches
            n_epochs          (int)  : Number of epochs (iterations of all minibatches)
            lmbda             (float): Ridge parameter lambda
            method            (str)  : Regreesion method of choice (either OLS or Ridge)
            check_equality    (Bool) : True or False -> if you want to compare with matrix inversion regression

        returns:
            beta           (np.array):   Array of beta parameters in linear model
        """
        n = self.n
        t0, t1 = 5, 10
        beta0 = self.beta0
        DM = self.DM
        y = self.y.reshape(n,1)
        v_prev = 0
        m_ = int(n / M)              # Number of minibatches
        def learning_rate(t):
            return t0 / (t1 + t)

        if method=="OLS":
            beta = np.zeros_like(beta0)
            for i, element in enumerate(beta0):
                beta[i] = element

            with np.printoptions(precision=3):
                print(f"Performing OLS regression using SGD with initial values = " ,(*beta0))

            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                m_prev = 0
                s_prev = 0
                t = 0
                mse =[]
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        t += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        mhat = m / (1 - avg_time1 ** t)
                        shat = s / (1 - avg_time2 ** t)

                        v = v_prev * gamma + eta * mhat / (np.sqrt(shat) + delta)
                        beta -= v
                        m_prev = m
                        s_prev = s
                        v_prev = v


            elif ADAGRAD:
                gradC = []
                delta = 1e-8
                mse =[]
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC.append(2 / M * (DMi.T @ (DMi @ beta - yi)))
                        k = len(gradC)
                        G = sum(gradC[j]@gradC[j].T for j in range(k))
                        dim1, dim2 = np.shape(G)
                        I = np.eye(dim1, dim2)
                        v = v_prev * gamma + (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]
                        beta -= v




            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        s = avg_time * s_prev + (1 - avg_time) * g ** 2
                        v = v_prev * gamma + eta * g / np.sqrt(s + delta)
                        beta -= v
                        v_prev = v
                        s_prev = s

            else:
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch*m_ + i)
                        DMi = DM[k*M:k*M + M]
                        yi = y[k*M:k*M + M]
                        gradC = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        v = v_prev * gamma + eta * gradC
                        beta -= v

            if check_equality:
                ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y
                np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)


        if method=="Ridge":
            beta = np.zeros_like(beta0)
            for i, element in enumerate(beta0):
                beta[i] = element

            with np.printoptions(precision=3):
                print(f"Performing Ridge regression with initial values = " ,(*beta0))

            if ADAM:
                delta = 1e-8
                avg_time1 = 0.9
                avg_time2 = 0.99
                m_prev = 0
                s_prev = 0
                t = 0
                mse =[]
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        t += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        mhat = m / (1 - avg_time1 ** t)
                        shat = s / (1 - avg_time2 ** t)
                        v = v_prev * gamma + eta * mhat / (np.sqrt(shat) + delta)
                        beta -= v
                        m_prev = m
                        s_prev = s
                        v_prev = v
                        mse.append(self.MSE(self.y,self.DM@beta))


            elif ADAGRAD:
                gradC = []
                delta = 1e-8
                mse =[]
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC.append(2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta)
                        k = len(gradC)
                        G = sum(gradC[j]@gradC[j].T for j in range(k))
                        dim1, dim2 = np.shape(G)
                        I = np.eye(dim1, dim2)
                        v = v_prev * gamma + (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * gradC[-1]
                        beta -= v
                        v_prev = v
                        mse.append(self.MSE(self.y,self.DM@beta))


            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                mse = []
                j =0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        s = avg_time * s_prev + (1 - avg_time) * g ** 2
                        v = v_prev * gamma + eta * g / np.sqrt(s + delta)
                        beta -= v
                        v_prev =  v
                        s_prev = s
                        mse.append(self.MSE(self.y,self.DM@beta))


            else:
                mse = []
                j = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        j += 1
                        mse.append(self.MSE(self.y,self.DM@beta))
                        k = np.random.randint(m_-1)              # Remove the upper m, since we add k*M + M meaning it'd crash
                                                                # if we by chance choose k = max(m)
                        eta = learning_rate(epoch*m_ + i)
                        DMi = DM[k*M:k*M + M]
                        yi = y[k*M:k*M + M]
                        XT_X = DMi.T @ DMi
                        dim1, dim2 = np.shape(XT_X)
                        I = np.eye(dim1, dim2)
                        gradC = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        v = v_prev * gamma + eta * gradC
                        beta -= v



            if check_equality:
                XT_X = DM.T @ DM
                ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y
                np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)

        return beta,mse


if __name__ == "__main__":
    sns.set_theme()
    np.random.seed(2001)
    x = np.linspace(0,1,100)
    inst = GD(x, 2)
    x = np.linspace(0,1000,1000)
    designmatrix = inst.DM
    ydata = inst.y



if False:    # RIDGE values

    ridge_beta1,mse_PlainGD        = inst.PlainGD(method="Ridge")
    ridge_beta2,mse_MGD            = inst.MGD(method="Ridge")
    ridge_beta3,mse_SGD            = inst.SGD(method="Ridge")
    ridge_beta4,mse_SMGD           = inst.SMGD(method="Ridge")
if False:   # RIDGE models
    model1_RIDGE = designmatrix @ ridge_beta1
    mse1_ols = inst.MSE(ydata, model1_ols)
    model2_RIDGE = designmatrix @ ridge_beta2
    mse2_ols = inst.MSE(ydata,model2_ols)
    model3_RIDGE = designmatrix @ ridge_beta3
    mse3_ols = inst.MSE(ydata,model3_ols)
    model4_RIDGE = designmatrix @ ridge_beta4
    mse4_ols = inst.MSE(ydata,model4_ols)
if False: # This code prints out all MSE for various eta values for different lmbda values ranging from [1E-1 - 1E-6]
    lmbdaa = np.array((0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001))
    mse_1 = np.zeros(len(lmbdaa))
    for i, lmbda_ in enumerate(lmbdaa):
        beta, mse= inst.PlainGD(eta=0.1,lmbda=lmbda_, method="Ridge")
        mse_1[i] = inst.MSE(ydata,designmatrix@beta)

    print(f"MSE of different lambda values[eta=0.01]: {mse_1}")




if False:# This code prints out all MSE for various lmbda values for different eta values ranging from [1E-1 - 1E-6]
        etaa = np.array((0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001))
        mse_2 = np.zeros(len(etaa))
        for i, etaaa in enumerate(etaa):
            beta,mse = inst.PlainGD(eta =etaaa,lmbda=0.1,method="Ridge")
            mse_2[i] = inst.MSE(ydata,designmatrix@beta)
        print(f"MSE of different eta values[lmbda=0.01]: {mse_2}")

if True: #Heatmap for mse, lmbda and eta
    lmbdaa = np.array((0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001))
    etaa = np.array((0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001))
    mse_1 = np.zeros((len(lmbdaa), len(etaa)))
    for i, lmbda_ in enumerate(lmbdaa):
        for j, etaaa in enumerate(etaa):
            beta,mse = inst.PlainGD(eta =etaaa,lmbda=lmbda_,method="Ridge")
            mse_1[i][j] = inst.MSE(ydata,designmatrix@beta)
    maxvalue_1 = np.max(mse_1)
    #Creating dataframe
    df1 = pd.DataFrame(data=mse_1[:,:],
                        index= etaa,
                         columns= lmbdaa)

    #Plotting heatmap for Optimal lambda and eta
    sns.heatmap(df1,annot=True,fmt='.2g')
    plt.title("Optimal hyperparamteres $\lambda$ and $\eta$")
    plt.xlabel("Learning rate $\eta$")
    plt.ylabel("Ridge parameter $\lambda$")
    plt.savefig("/Users/fuaddadvar/fys-STK4155/partA/Heatmap_MSE_lmbda_eta.pdf")
    plt.show()
