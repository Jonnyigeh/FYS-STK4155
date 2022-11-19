import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    def ADAM(self, g, i, m_prev=0, s_prev=0):
        avg_time1 = 0.9
        avg_time2 = 0.99
        m = avg_time1 * m_prev + (1 - avg_time1) * g
        s = avg_time2 * s_prev + (1 - avg_time2) * g **2
        mhat = m / (1 - avg_time1 ** i)
        shat = s / (1 - avg_time2 ** i)

        return m, s, mhat, shat

    def RMSprop(self, g, s_prev=0):
        avg_time = 0.9
        s = avg_time * s_prev + (1 - avg_time) * g ** 2

        return s

    def ADAGRAD(self, g, G):
        G += g @ g.T
        dim1, dim2 = np.shape(G)
        I = np.eye(dim1, dim2)

        return G, I

    def SGD(self, M = 10, n_epochs = 50, lmbda = 0.001):
        t0, t1 = 5, 50
        delta = 1e-8
        def learning_rate(t):
            return t0 / (t1 + t)

        n = self.n
        beta0 = self.beta0
        DM = self.DM
        y = self.y.reshape(n,1)
        n_batches = int(n / M)                                  # Number of minibatches
        def reset_beta():                                       # Resets the beta values (to prevent different starting points for the vairous methods)
            beta = np.zeros_like(beta0)
            for i, element in enumerate(beta0):
                beta[i] = element
            return beta

        mse_adam = []
        r2_adam = []
        beta = reset_beta()
        m_prev = 0
        s_prev = 0
        t = 1

        for epoch in range(n_epochs):
            for i in range(n_batches):
                # Finding the MSE in each step, appending to list for later plotting
                mse_adam.append(self.MSE(self.y, self.DM@beta))
                r2_adam.append(self.R2(self.y, self.DM@beta))
                # Batching the data, and finding the learning rate
                k = np.random.randint(n_batches-1)
                eta = learning_rate(epoch * n_batches + i)
                DMi = DM[k*M:k*M + M]
                yi = y[k*M:k*M + M]
                # Evaluating the gradient, finding ADAM params and update beta
                g = (2 / M) * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                m, s, mhat, shat = self.ADAM(g,t, m_prev, s_prev)
                beta -= eta * mhat / (np.sqrt(shat) + delta)
                t+=1
                m_prev = m
                s_prev = s


        mse_adag = []
        r2_adag = []
        beta = reset_beta()
        G_prev = 0

        for epoch in range(n_epochs):
            for i in range(n_batches):
                mse_adag.append(self.MSE(self.y, self.DM@beta))
                r2_adag.append(self.R2(self.y, self.DM@beta))

                k = np.random.randint(n_batches-1)
                eta = learning_rate(epoch * n_batches + i)
                DMi = DM[k*M:k*M + M]
                yi = y[k*M:k*M + M]
                g = (2 / M) * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                G, I = self.ADAGRAD(g, G_prev)
                beta -= (eta * 1 / np.sqrt(np.diag(G + delta * I))).reshape(*beta.shape) * g
                G_prev = G


        mse_rms = []
        r2_rms = []
        beta = reset_beta()
        s_prev = 0

        for epoch in range(n_epochs):
            for i in range(n_batches):
                mse_rms.append(self.MSE(self.y, self.DM@beta))
                r2_rms.append(self.R2(self.y, self.DM@beta))

                k = np.random.randint(n_batches-1)
                eta = learning_rate(epoch * n_batches + i)
                DMi = DM[k*M:k*M + M]
                yi = y[k*M:k*M + M]
                g = (2 / M) * (DMi.T @ (DMi @ beta - yi)) + 2 * lmbda * beta
                s = self.RMSprop(g,s_prev)
                beta -= (eta * g / (np.sqrt(s) + delta))
                s_prev = s

        # Visualization of the rates of convergence for these various adaptive learning rate methods
        i = np.arange(0, len(mse_adam), 1)
        fig = plt.figure()
        axs = fig.add_subplot(1,1,1)
        axs.set_ylabel("MSE", fontsize=10)
        axs.set_xlabel("Number of iterations", fontsize=10)
        axs.plot(i, mse_adam/np.max(mse_adam), i, mse_adag/np.max(mse_adam), i, mse_rms/np.max(mse_adam))
        axs.set_title("MSE for the adaptive learning rate methods")
        axs.legend(["ADAM", "ADAGRAD", "RMSprop"])
        fig.tight_layout()
        plt.savefig("rate_of_convergence_ALR_methods.pdf")
        plt.show()


if __name__=="__main__":
    sns.set_theme()
    np.random.seed(0)
    x = np.linspace(0,1,100)
    inst = GD(x, 2)
    inst.SGD()
