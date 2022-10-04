import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from random import random, seed
from sklearn.utils import resample

seed(2020)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def create_X(x,y,n):
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

def generate_linear_model(DM, Z, dim_1 = False, Ridge = False, Lasso = False):
    """Generates the beta's for a linearmodel fit
            using matrix inversion of the design matrix.

    Args:
        DM = Design matrix
        Z = Output data from a function on a grid

    Returns:
        beta = Dictionary with beta values for polynomial fit of
                    degree 1 -> 5, with degree as key.
        with dim_1 = True
            beta = single array of beta values for the desired polynomial
    """

    if dim_1:
        """
        True if you want the function to return a single
            np.array with betas corresponding to a single polynomial fit,
                where polynomial degree is decided by the Design Matrix DM.
        """
        if Ridge:
            """
            Performs Ridge Regression and
                return the coefficients in the linear fit (betas) for
                    100 lambdas in the interval log[-4, 4]
            """
            nlambdas = 100
            lmb = np.logspace(-6,6, nlambdas)
            beta = np.zeros((100, len(DM[1])))
            for i in range(nlambdas):
                hessian = DM.T.dot(DM)
                width, height = np.shape(hessian)
                I = np.eye(width, height)
                beta[i] = np.linalg.inv(hessian + lmb[i] * I).dot(DM.T).dot(Z.ravel())

            return beta, lmb

        if Lasso:
            """
            Performs Lasso linear_regression and
                returns the coefficients for the linear fits (betas) for
                    100 lambdas in the interval log[-4, 4]
            """
            nlambdas = 100
            lmb = np.logspace(-4,4, nlambdas)
            beta = np.zeros(100)
            for i in range(nlambdas):
                lasso_reg = Lasso(lmb)
                lasso_reg.fit(DM,Z)
                beta[i] = lasso_reg.coef_

            return beta

        else:

            beta = np.linalg.inv(DM.T.dot(DM)).dot(DM.T).dot(Z.ravel())

            return beta
    else:
        beta = {}
        for k in range(1, degree+1):
            beta[k] = np.linalg.inv(DM[k].T.dot(DM[k])).dot(DM[k].T).dot(Z.ravel())

        return beta

def generate_DM(X,Y):
    """Generates a dictionary of the design matrix
            for polynomial degree 1-> 5, where degree is key.

    Args:
        X = grid of x-datapoints
        Y = grid of y-datapoints

    Returns:
        DM = Dictionary of design matrices of polynomial fit of
                    k, where k is the key.
    """
    DM = {}
    for k in range(1,degree+1):
        DM[k] = create_X(X.ravel(),Y.ravel(),n=k)

    return DM

def generate_MSE_R2(DM, beta, Z, dim_1 = False):
    """Generates the Mean Squared Error (MSE), and the R^2 (R2)
            for the polynomial fits given by the linear regression model
                by DM and beta.

    Args:
        DM = Design matrix used for the polynomial fit
        beta = Beta values for the polynomial fit
        Z = Function values that are used to fit

    Returns:
        MSE = Dictionary of MSE, with key indicating polynomial degree
        R2 = Dictionary of R2, with key indicating polynomial degree

    """
    if dim_1:
        Z_mean = np.mean(Z.ravel())
        Z_tilde = DM.dot(beta)
        Z_tilde_mean = np.mean(Z_tilde.ravel())
        n = len(Z_tilde)
        MSE = 1 / n * sum((Z.ravel()[i] - Z_tilde.ravel()[i]) **2 for i in range(len(Z.ravel())))
        R_squared = 1 - ( sum((Z.ravel()[i] - Z_tilde.ravel()[i]) **2 for i in range(len(Z.ravel()))) /
                                sum((Z.ravel()[i] - Z_mean) **2 for i in range(len(Z.ravel()))) )

        Var_Z =  np.mean( np.var(( Z_tilde.ravel())))
        Bias_Z = np.mean( (Z.ravel() - Z_tilde_mean) **2)

        return MSE, R_squared, Var_Z, Bias_Z

    else:
        Z_mean = np.mean(Z.ravel())
        Z_tilde = {}
        MSE = {}
        R_squared = {}
        for k in range(1, degree+1):
            Z_tilde[k] = DM[k].dot(beta[k])
            n = len(Z_tilde[k])
            MSE[k] = 1 / n * sum((Z.ravel()[i] - Z_tilde[k].ravel()[i]) **2 for i in range(len(Z.ravel())))
            R_squared[k] = 1 - ( sum((Z.ravel()[i] - Z_tilde[k].ravel()[i]) **2 for i in range(len(Z.ravel()))) /
                                    sum((Z.ravel()[i] - Z_mean) **2 for i in range(len(Z.ravel()))) )
        Var_Z = {}
        Bias_Z = {}
        for k in range(1,degree+1):
            Z_tilde_mean = np.mean(Z_tilde[k].ravel())
            Var_Z[k] = 1 / n * sum( (Z_tilde[k].ravel()[i] - Z_tilde_mean) **2 for i in range(len(Z.ravel())) )
            Bias_Z[k] = 1 / n * sum( (Z.ravel()[i] - Z_tilde_mean) **2 for i in range(len(Z.ravel())) )
        return MSE, R_squared, Var_Z, Bias_Z

if __name__ == "__main__":
    n = 40
    degree = 5
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y)
    xnoise = np.random.randn(len(x)) * 0.25
    ynoise = np.random.randn(len(x)) * 0.25
    Xnoise, Ynoise = np.meshgrid(xnoise, ynoise)
    Z = FrankeFunction(X,Y) + (Xnoise + Ynoise)
    DM = generate_DM(X,Y)
    DM_train = {}
    DM_test = {}
    for k in range(1,degree+1):
        DM_train[k], DM_test[k], Z_train, Z_test = train_test_split(DM[k], Z.ravel(), test_size=0.33)

    """
    Linear regression part
    """

    reg = LinearRegression().fit(DM_train[5], Z_train)

    model = reg.predict(DM_test[5])
    beta = generate_linear_model(DM_train, Z_train)
    MSE, R2, Var, Bias = generate_MSE_R2(DM_train, beta, Z_train)

    """
    Ridge
    """
    # DM_train_ridge, DM_test_ridge, Z_train_ridge, Z_test_ridge = train_test_split(DM[5], Z.ravel(), test_size=0.33)
    # beta_ridge, lmb_ridge = generate_linear_model(DM_train_ridge, Z_train_ridge, dim_1 = True, Ridge = True)
    # MSE_ridge = np.zeros(100)
    # for i in range(len(MSE_ridge)):
    #     MSE_ridge[i], R2, Var, Bias = generate_MSE_R2(DM_test_ridge, beta_ridge[i], Z_test_ridge, dim_1 = True)
    """
    Lasso
    """

    """
    Bootstrap part
    """

    bootstrap_n = 100
    MSE_new_sum = np.zeros((degree, bootstrap_n))
    Var_Z_sum = np.zeros_like(MSE_new_sum)
    Bias_Z_sum = np.zeros_like(MSE_new_sum)

    for k in range(1,degree+1):
        for i in range(bootstrap_n):
            DM_new, Z_new = resample(DM_train[k],Z_train)
            beta_new = generate_linear_model(DM_new, Z_new, dim_1 = True)
            MSE_new_sum[k-1, i], R2_new, Var_Z_sum[k-1, i], Bias_Z_sum[k-1, i] = generate_MSE_R2(DM_test[k], beta_new, Z_test, dim_1 = True)

    
    
    
    
    
    MSE_boot = np.ones(degree)
    Var_boot = np.ones_like(MSE_boot)
    Bias_boot = np.ones_like(MSE_boot)
    for i in range(degree):
        MSE_boot[i] = np.mean(MSE_new_sum[i])
        Var_boot[i] = np.mean(Var_Z_sum[i])
        Bias_boot[i] = np.mean(Bias_Z_sum[i])

    p = np.arange(1,degree+1,1)
    plt.plot(p, MSE_boot, p, Var_boot, p, Bias_boot)
    plt.legend(["MSE", "Var", "Bias"])
    plt.show()
    breakpoint()
    #
    # x = np.linspace(1,21,21)
    # fig, axs = plt.subplots(2, 2)
    # b = np.linspace(0,21,21)
    # axs[0][0].plot([1,2,3,4,5], [MSE[1], MSE[2], MSE[3], MSE[4], MSE[5]], "-o")
    # axs[0][0].plot([1,2,3,4,5], [R2[1], R2[2], R2[3], R2[4], R2[5]], "-o")
    # axs[0][0].legend(["MSE", "R2"])
    # axs[0][1].scatter(x[0:3], beta[1])
    # axs[1][0].scatter(x[0:10], beta[3])
    # axs[1][1].scatter(x, beta[5])
    # axs[0][1].legend(["Beta for polynomial degree 1"])
    # axs[1][0].legend(["Beta for polynomial degree 3"])
    # axs[1][1].legend(["Beta for polynomial degree 5"])
    # plt.show()
    # plt.plot([1,2,3,4,5], [MSE[1], MSE[2], MSE[3], MSE[4], MSE[5]], "-o")
    # plt.plot([1,2,3,4,5], [MSE_2[1], MSE_2[2], MSE_2[3], MSE_2[4], MSE_2[5]], "k-o")
    # breakpoint()
