import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from random import random, seed
from sklearn.utils import resample

seed(2020)

def f(x):
    return x **2

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

def OLS(DM, Z):
    beta = np.linalg.pinv(DM.T @ DM) @ DM.T @ Z.ravel()
    model = beta

    return model

def Ridge(DM, Z, lmb):
    hessian = DM.T @ DM
    dim1, dim2 = np.shape(hessian)
    I = np.eye(dim1, dim2)
    beta = np.linalg.pinv(hessian + lmb * I) @ DM.T @ Z.ravel()
    model = beta

    return model

def Lasso(DM, Z, lmb):
    lasso_reg = Lasso(lmb)
    lasso_reg.fit(DM,Z)
    model = lasso_reg.predict(DM)

    return model, lasso_reg.coef_

def MSE(Z, Z_model):
    n = len(Z_model)
    mse = np.mean( (Z.ravel() - Z_model) **2 )

    return mse

def R2(Z, Z_model):
    n = len(Z_model)
    Z_mean = np.mean(Z.ravel())
    r2 = 1 - np.sum((Z.ravel() - Z_model) ** 2) / np.sum((Z.ravel() - Z_mean) ** 2)

    return r2

def Var(model):
    variance = np.mean( np.var(model) )

    return variance

def Bias(data, model):
    bias = np.mean( (data.ravel() - np.mean(model)) ** 2)

    return bias

def mean_scale(data, mean_data):
    new_data = ( data - mean_data )

    return new_data

def plotin3D(x,y,z):
    """Generates a 3D surfaceplot of the data given by Z

    args:
        x: meshgrid of x points
        y: meshgrid of y points
        z: meshgrid of datapoints (z)
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(x,y,z)
    plt.show()


def bootstrap_OLS(train_data, test_data, niteration):
    """ Performs bootstrap
    args:
        train_data    (tuple): tuple of X and Y training data
        test_data     (tuple): tuple of X and Y test data
        niteration      (int): Number of desidered bootstrap iterations

    returns:
        MSE
        Var
        Bias
    """
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    X_train = mean_scale(X_train, np.mean(X_train))
    X_test = mean_scale(X_test, np.mean(X_train))
    Y_train = mean_scale(Y_train, np.mean(Y_train))
    # Y_test = mean_scale(Y_test, np.mean(Y_train))

    mse = np.zeros(niteration)
    var = np.zeros_like(mse)
    bias = np.zeros_like(mse)
    for i in range(niteration):
        new_X, new_Y = resample(X_train, Y_train)
        model = OLS(new_X, new_Y)
        mse[i] = MSE(Y_test, X_test @ model)
        var[i] = Var(X_test @ model)
        bias[i] = Bias(Y_test, X_test @ model)

    return np.mean(mse), np.mean(var), np.mean(bias)

if __name__ == "__main__":
    n = 40
    degree = 5
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y)
    sigma2 = 0.05
    xnoise = np.random.randn(len(x)) * sigma2
    ynoise = np.random.randn(len(x)) * sigma2
    Xnoise, Ynoise = np.meshgrid(xnoise, ynoise)
    Z = FrankeFunction(X,Y) + (Xnoise + Ynoise)
    Z_onedim = f(x) + xnoise

    # Sklearn reference model
    var_ypred = np.ones(5)
    bias_ypred = np.ones(5)
    mse_ypred = np.ones(5)
    for i in range(degree):
        DM_ref = create_X(X,Y,i+1)
        DM_ref_train, DM_ref_test, Z_train, Z_test = train_test_split(DM_ref, Z.ravel(), test_size=0.33)
        lin_reg = LinearRegression(fit_intercept=False)
        lin_reg.fit(DM_ref_train, Z_train)
        ypredict = lin_reg.predict(DM_ref_test)
        var_ypred[i] = Var(ypredict)
        bias_ypred[i] = Bias(Z_test, ypredict)
        mse_ypred[i] =  MSE(Z_test, ypredict)
    breakpoint()


    # Part B
    degree = 25                 # Maximum degree of polynomial fit
    R2_ols = np.zeros(degree)
    MSE_ols = np.zeros_like(R2_ols)
    MSE_ols_train = np.zeros_like(R2_ols)
    Var_ols = np.zeros_like(R2_ols)
    Bias_ols = np.zeros_like(R2_ols)
    model_ols = {}
    DM_ols = {}
    Var_beta = {}

    for i in range(len(MSE_ols)):
        DM = create_X(X, Y, i+1)
        DM_train, DM_test, Z_train, Z_test = train_test_split(DM, Z.ravel(), test_size=0.33) # splits our data
        DM_ols[i+1] = DM_test
        Z_train = mean_scale(Z_train) # Mean scaling of the data
        Z_test = mean_scale(Z_test) # Same as above
        model_ols[i+1] = OLS(DM_train, Z_train) # Produces betas for our linear regression model
        MSE_ols[i] = MSE(Z_test, DM_test @ model_ols[i+1])
        MSE_ols_train[i] = MSE(Z_train, DM_train @ model_ols[i+1])
        R2_ols[i] = R2(Z_test, DM_test @ model_ols[i+1])
        Var_beta[i+1] = sigma2 * np.linalg.pinv(DM_test.T @ DM_test).diagonal() # Gives variance in betas (model_ols)

        Var_ols[i] = Var(DM_test @ model_ols[i+1])
        Bias_ols[i] = Bias(Z_test, DM_test @ model_ols[i+1])

    # Part c
    boot_ols_mse = np.zeros(3)
    boot_ols_var = np.zeros(3)
    boot_ols_bias = np.zeros(3)
    for i, j in enumerate([1, 3, 5]):
        DM = create_X(X,Y, j)
        DM_train, DM_test, Z_train, Z_test = train_test_split(DM, Z.ravel(), test_size=0.33)
        # Z_train = mean_scale(Z_train)
        # Z_test = mean_scale(Z_test)
        boot_ols_mse[i], boot_ols_var[i], boot_ols_bias[i] = bootstrap_OLS((DM_train, Z_train), (DM_test, Z_test), niteration=100)

    # x1 = np.linspace(0, np.)
    # x2 =
    # x3 =
    # x4 =
    breakpoint()


    # Var_ols[i] = Var(DM_test @ model_ols)
    # Bias_ols[i] = Bias(Z_test, DM_test @ model_ols)
