import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


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

    def PlainGD(self, eta = 0.02, eps = 10 ** (-8), lmbda = 0.02, method= "OLS", RMSprop=False, ADAGRAD=True, ADAM=False, check_equality = False):
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

                while np.linalg.norm(g) > eps:
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    m = m / (1 - avg_time1 ** i)
                    s = s / (1 - avg_time2 ** i)

                    beta -= eta * m / (np.sqrt(s) + delta)
                    m_prev = m
                    s_prev = s
            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)))
                delta = 1e-8
                G = gradC[0]@gradC[0].T
                dim1, dim2 = np.shape(G)
                I = np.eye(dim1, dim2)

                beta = beta0 - eta * 1 / np.sqrt(np.diag(G + delta * I))
                
                while np.linalg.norm(gradC) > eps:
                    gradC.append(2 / n * (DM.T @ (DM @ beta - y)))
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    beta -= eta * 1 / np.sqrt(np.diag(G + delta * I))

            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                s_prev = s
                beta = beta0 - eta * g / np.sqrt(s + delta)

                while np.linalg.norm(g) > eps:
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    s = avg_time * s_prev + (1 - avg_time) * g ** 2
                    beta -= eta * g / np.sqrt(s + delta)
                    s_prev = s
            else:
                breakpoint()
                gradC = 2 / n * (DM.T @ (DM @ beta0 - y))
                beta = beta0 - eta * gradC
                while np.linalg.norm(gradC) > eps:
                    gradC = 2 / n * (DM.T @ (DM @ beta - y))
                    beta -= eta * gradC
            
            ref_beta = np.linalg.pinv(DM.T @ DM) @ DM.T @ y
            


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

                while np.linalg.norm(g) > eps:
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    m = m / (1 - avg_time1 ** i)
                    s = s / (1 - avg_time2 ** i)

                    beta -= eta * m / (np.sqrt(s) + delta)
                    m_prev = m
                    s_prev = s



            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta)
                delta = 1e-8
                G = gradC[0]@gradC[0].T
                beta = beta0 - eta * 1 / np.sqrt(np.diag(G + delta * I))

                while np.linalg.norm(gradC) > eps:
                    gradC.append(2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta)
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    beta -= eta * 1 / np.sqrt(np.diag(G + delta * I))


            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                gradC = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                beta -= eta * g / np.sqrt(s + delta)

                while np.linalg.norm(gradC) > eps:
                    g = 2 / n * (DM.T @ (DM @ beta - y))
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
                while np.linalg.norm(gradC) > eps:
                    gradC = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    beta -= eta * gradC

            ref_beta = np.linalg.pinv(DM.T @ DM + lmbda * I ) @ DM.T @ y


        if check_equality:
            np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)


        return beta
        breakpoint()
    def MGD(self, eta = 0.01, eps = 10 ** (-8), gamma = 0.9, lmbda = 0.01, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):
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

                while np.linalg.norm(g) > eps:
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y))
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    m = m / (1 - avg_time1 ** i)
                    s = s / (1 - avg_time2 ** i)
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
                beta = beta0 - eta * 1 / np.sqrt(np.diag(G + delta * I))

                while np.linalg.norm(gradC) > eps:
                    gradC.append(2 / n * (DM.T @ (DM @ beta - y)))
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    v = gamma * v_prev + eta * 1 / np.sqrt(np.diag(G + delta * I))
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

                while np.linalg.norm(gradC) > eps:
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

                while np.linalg.norm(gradC) > eps:
                    gradC = 2 / n * (DM.T @ (DM @ beta - y))
                    v = v_prev * gamma + eta * gradC
                    beta -= v
                    v_prev = v

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

                while np.linalg.norm(g) > eps:
                    i += 1
                    g = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    m = avg_time1 * m_prev + (1 - avg_time1) * g
                    s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                    m = m / (1 - avg_time1 ** i)
                    s = s / (1 - avg_time2 ** i)
                    v = gamma * v_prev + eta * m / (np.sqrt(s) + delta)
                    beta -= v
                    m_prev = m
                    s_prev = s
                    v_prev = v



            elif ADAGRAD:
                gradC = []
                gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta)
                delta = 1e-8
                G = gradC[0]@gradC[0].T
                v_prev = 0
                beta = beta0 - eta * 1 / np.sqrt(np.diag(G + delta * I))

                while np.linalg.norm(gradC) > eps:
                    gradC.append(2 / n * (DM.T @ (DM @ beta0 - y)) + 2 * lmbda * beta)
                    k = len(gradC)
                    G = sum(gradC[i]@gradC[i].T for i in range(k))
                    dim1, dim2 = np.shape(G)
                    I = np.eye(dim1, dim2)
                    v = gamma * v_prev + eta * 1 / np.sqrt(np.diag(G + delta * I))
                    beta -= v
                    v_prev = v

            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                v_prev = 0
                g = 2 / n * (DM.T @ (DM @ beta0 - y))
                s = avg_time * s_prev + (1 - avg_time) * g ** 2
                beta -= eta * g / np.sqrt(s + delta)

                while np.linalg.norm(gradC) > eps:
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

                while np.linalg.norm(gradC) > eps:
                    gradC = 2 / n * (DM.T @ (DM @ beta - y)) + 2 * lmbda * beta
                    v = v_prev * gamma + eta * gradC
                    beta -= v
                    v_prev = v




            ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y

        if check_equality:
            np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)


        return beta

    def SGD(self, M = 50, n_epochs = 100, lmbda = 0.001, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):
        """Performs linear regression using Stochastic Gradient Descent
                    using tunable learning rate. (see function *learning rate*)

        args:
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

                for epoch in range(n_epochs):
                    for i in range(m_):
                        t += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        m = m / (1 - avg_time1 ** t)
                        s = s / (1 - avg_time2 ** t)

                        beta -= eta * m / (np.sqrt(s) + delta)
                        m_prev = m
                        s_prev = s


            elif ADAGRAD:
                gradC = []
                delta = 1e-8
                for epoch in range(n_epochs):
                    for i in range(m_):
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC.append(2 / M * (DMi.T @ (DMi @ beta - yi)))
                        k = len(gradC)
                        G = sum(gradC[j]@gradC[j].T for j in range(k))
                        dim1, dim2 = np.shape(G)
                        I = np.eye(dim1, dim2)
                        beta -= eta * 1 / np.sqrt(np.diag(G + delta * I))


            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        s = avg_time * s_prev + (1 - avg_time) * g ** 2
                        beta -= eta * g / np.sqrt(s + delta)
                        s_prev = s

            else:
                for epoch in range(n_epochs):
                    for i in range(m_):
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
                for epoch in range(n_epochs):
                    for i in range(m_):
                        t += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        m = m / (1 - avg_time1 ** t)
                        s = s / (1 - avg_time2 ** t)

                        beta -= eta * m / (np.sqrt(s) + delta)
                        m_prev = m
                        s_prev = s


            elif ADAGRAD:
                gradC = []
                delta = 1e-8
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
                        beta -= eta * 1 / np.sqrt(np.diag(G + delta * I))



            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        s = avg_time * s_prev + (1 - avg_time) * g ** 2
                        beta -= eta * g / np.sqrt(s + delta)
                        s_prev = s


            else:
                for epoch in range(n_epochs):
                    for i in range(m_):
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
                XT_X = DM.T @ DM
                ref_beta = np.linalg.pinv(XT_X + lmbda * I ) @ DM.T @ y
                np.testing.assert_allclose(beta, ref_beta, rtol=1e-3, atol=1e-3)

        return beta

    def SMGD(self, M = 50, n_epochs = 100, gamma=0.9, lmbda = 0.001, method= "OLS", RMSprop=False, ADAGRAD=False, ADAM=False, check_equality = False):
        """Performs linear regression using Stochastic Gradient Descent with momentum
                and a tunable learning rate. (see function learning rate).

        args:
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
                for epoch in range(n_epochs):
                    for i in range(m_):
                        t += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi))
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        m = m / (1 - avg_time1 ** t)
                        s = s / (1 - avg_time2 ** t)

                        v = v_prev * gamma + eta * m / (np.sqrt(s) + delta)
                        beta -= v
                        m_prev = m
                        s_prev = s
                        v_prev = v


            elif ADAGRAD:
                gradC = []
                delta = 1e-8
                for epoch in range(n_epochs):
                    for i in range(m_):
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        gradC.append(2 / M * (DMi.T @ (DMi @ beta - yi)))
                        k = len(gradC)
                        G = sum(gradC[j]@gradC[j].T for j in range(k))
                        dim1, dim2 = np.shape(G)
                        I = np.eye(dim1, dim2)
                        v = v_prev * gamma + eta * 1 / np.sqrt(np.diag(G + delta * I))
                        beta -= v




            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
                for epoch in range(n_epochs):
                    for i in range(m_):
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
                for epoch in range(n_epochs):
                    for i in range(m_):
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
                for epoch in range(n_epochs):
                    for i in range(m_):
                        t += 1
                        k = np.random.randint(m_-1)
                        eta = learning_rate(epoch * m_ + i)         # the addition of i in the function call means that the learning
                        DMi = DM[k*M:k*M + M]                      # rate is not constant through the minibatch, but decays somewhat.
                        yi = y[k*M:k*M + M]
                        g = 2 / M * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                        m = avg_time1 * m_prev + (1 - avg_time1) * g
                        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
                        m = m / (1 - avg_time1 ** t)
                        s = s / (1 - avg_time2 ** t)
                        v = v_prev * gamma + eta * m / (np.sqrt(s) + delta)
                        beta -= v
                        m_prev = m
                        s_prev = s
                        v_prev = v



            elif ADAGRAD:
                gradC = []
                delta = 1e-8
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
                        v = v_prev * gamma + eta * 1 / np.sqrt(np.diag(G + delta * I))
                        beta -= v
                        v_prev = v


            elif RMSprop:
                avg_time = 0.9
                delta = 1e-8
                s_prev = 0
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


            else:
                for epoch in range(n_epochs):
                    for i in range(m_):
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

        return beta


if __name__ == "__main__":
    sns.set_theme()
    np.random.seed(2001)
    x = np.linspace(0,1,100)
    inst = GD(x, 2)
    
    designmatrix = inst.DM
    ydata = inst.y

    
    ols_beta1 = inst.PlainGD()
    ols_beta2 = inst.MGD()
    ols_beta3 = inst.SGD()
    ols_beta4 = inst.SMGD()

    ridge_beta1 = inst.PlainGD(method="Ridge")
    ridge_beta2 = inst.MGD(method="Ridge")
    ridge_beta3 = inst.SGD(method="Ridge")    
    ridge_beta4 = inst.SMGD(method="Ridge")

#OLS
    model1_ols = designmatrix @ ols_beta1
    mse1_ols = inst.MSE(ydata, model1_ols)
    
    model2_ols = designmatrix @ ols_beta2
    mse2_ols = inst.MSE(ydata,model2_ols)

    model3_ols = designmatrix @ ols_beta3
    mse3_ols = inst.MSE(ydata,model3_ols)

    model4_ols = designmatrix @ ols_beta4
    mse4_ols = inst.MSE(ydata,model4_ols)

#RIDGE
    model1_ridge = designmatrix @ ridge_beta1
    mse1_ridge = inst.MSE(ydata, model1_ridge)
    
    model2_ridge = designmatrix @ ridge_beta2
    mse2_ridge = inst.MSE(ydata,model2_ridge)

    model3_ridge = designmatrix @ ridge_beta3
    mse3_ridge = inst.MSE(ydata,model3_ridge)

    model4_ridge = designmatrix @ ridge_beta4
    mse4_ridge = inst.MSE(ydata,model4_ridge)
    lrates = [""]

    print(f"MSE of GD: {mse1_ols}")
    
    
    #Plotting
    # plt.plot(x,ydata,label = "ydata with noise",linestyle ='*',color='m')
    # plt.plot(x,model1_ols, label = "Plain GD")
    # plt.plot(x,model3_ols,label ="Stochastic GD")
    # plt.plot(x,model4_ols,label="Stochastic GD with momentum")

    # plt.legend(loc="lower right",prop={'size': 8})
    # plt.show()
    
    





# # Load the example flights dataset and convert to long-form
# flights_long = sns.load_dataset("flights")
# flights = flights_long.pivot("month", "year", "passengers")

# # Draw a heatmap with the numeric values in each cell
# f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
