import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from random import random, seed
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sys import exit

seed(411)

def f(x):
    return x

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

    return lasso_reg.coef_

def MSE(Z, Z_model):
    mse = np.mean( (Z.ravel() - Z_model) **2)

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

def plotin3D(x,y,z, show=False):
    """Generates a 3D surfaceplot of the data given by Z

    args:
        x: meshgrid of x points
        y: meshgrid of y points
        z: meshgrid of datapoints (z)
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(x,y,z)
    if show:
        plt.show()


def bootstrap(X_train, Y_train, X_test, Y_test, method="OLS", lmbda=0, niteration=100):
    """ Performs bootstrap
    args:

    """
    X_train = mean_scale(X_train, np.mean(X_train))
    Y_train = mean_scale(Y_train, np.mean(Y_train))
    X_test = mean_scale(X_test, np.mean(X_train))
    Y_test = mean_scale(Y_test, np.mean(Y_train))

    model = np.zeros((np.shape(Y_test)[0], niteration))
    #sk_model = np.zeros((np.shape(Y_test)[0], niteration))
    if method=="OLS":
        for i in range(niteration):
            new_X, new_Y = resample(X_train, Y_train)       # Resamples the design matrix and the Z datapoints
            model[:,i] = X_test @ OLS(new_X, new_Y).ravel()
            # sk_model[:,i] = LinearRegression(fit_intercept=False).fit(new_X,new_Y).predict(X_test)

        # error2 = np.mean( np.mean((Y_test - sk_model)**2, axis=1, keepdims=True) )
        # bias2 = np.mean( (Y_test - np.mean(sk_model, axis=1, keepdims=True))**2 )
        # variance2 = np.mean( np.var(sk_model, axis=1, keepdims=True) )

        # return error2, variance2, bias2

    if method=="Ridge":
        for i in range(niteration):
            new_X, new_Y = resample(X_train, Y_train)
            model[:,i] = X_test @ Ridge(new_X, new_Y, lmbda)

    if method=="Lasso":
        for i in range(niteration):
            new_X, new_Y = resample(X_train, Y_train)
            model[:,i] = X_test @ Lasso(new_X, new_Y, lmbda)


    Y_test = Y_test.reshape(np.shape(Y_test)[0],1)
    error = np.mean( np.mean((Y_test - model)**2, axis=1, keepdims=True) )
    bias = np.mean( (Y_test - np.mean(model, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(model, axis=1, keepdims=True) )

    return error, variance, bias


if __name__ == "__main__":
    n = 40         # Number of datapoints
    if True:       # Setting up datastructures
        x = np.linspace(0,1,n)
        y = np.linspace(0,1,n)
        X,Y = np.meshgrid(x,y)
        sigma = 0.03
        Z = FrankeFunction(X,Y) + np.random.randn(n,n) * np.sqrt(sigma)
        Z_onedim = f(x) + np.random.randn(n) * np.sqrt(sigma)


    one_dim = False
    if one_dim:             # One dimensional test
        DM = np.ones((n,6))
        DM_eye = np.eye(n)
        DM[:,1] = x
        DM[:,2] = x **2
        DM[:,3] = x **3
        DM[:,4] = x **4
        DM[:,5] = x **5
        list_of_polydegree = [1,2,3,4,5]
        boot_ols_mse = np.zeros(len(list_of_polydegree))
        boot_ols_var = np.zeros_like(boot_ols_mse)
        boot_ols_bias = np.zeros_like(boot_ols_mse)
        for i, j in enumerate(list_of_polydegree):
            # DM = create_X(X,Y,j)
            DM_ = DM[:,:j]
            DM_train, DM_test, Z_train, Z_test = train_test_split(DM_, Z_onedim, test_size=0.33)
            boot_ols_mse[i], boot_ols_var[i], boot_ols_bias[i] = bootstrap_OLS(DM_train, Z_train, DM_test, Z_test, niteration=100)

        # DM_train, DM_test, Z_train, Z_test = train_test_split(DM[:,:3], Z_onedim, test_size=0.33, shuffle=False)
        # model_onedim = OLS(DM_train, Z_train)
        # lin_reg = LinearRegression(fit_intercept=False)
        # lin_reg.fit(DM_train, Z_train)
        # ypred = lin_reg.predict(DM_test)
        # new_model_onedim = DM_test @ model_onedim
        # print(Bias(Z_test, new_model_onedim))
        # print(Var(new_model_onedim))
        # print(MSE(Z_test, new_model_onedim))


    # Sklearn reference model

    ref_model = False
    if ref_model:          # Testing with sklearns model
        var_ypred = np.ones(5)
        bias_ypred = np.ones(5)
        mse_ypred = np.ones(5)
        for i in range(degree):
            DM_ref = create_X(X,Y,i+1)
            DM_ref_train, DM_ref_test, Z_train, Z_test = train_test_split(DM_ref, Z.ravel(), test_size=0.33)#, shuffle=False)
            lin_reg = LinearRegression(fit_intercept=False)
            lin_reg.fit(DM_ref_train, Z_train)
            ypredict = lin_reg.predict(DM_ref_test)
            var_ypred[i] = Var(ypredict)
            bias_ypred[i] = Bias(Z_test, ypredict)
            mse_ypred[i] =  MSE(Z_test, ypredict)

    # Part B
    part_b = False          # Performs part b (somewhat)
    if part_b:
        degree = 5                 # Maximum degree of polynomial fit
        R2_ols = np.zeros(degree)
        MSE_ols = np.zeros_like(R2_ols)
        MSE_ols_train = np.zeros_like(R2_ols)
        Var_ols = np.zeros_like(R2_ols)
        Bias_ols = np.zeros_like(R2_ols)
        model_ols = {}
        DM_ols = {}
        DM_ols_test = {}
        Var_beta = {}

        for i in range(len(MSE_ols)):
            DM = create_X(X, Y, i+1)
            DM_ols[i+1] = DM
            DM_train, DM_test, Z_train, Z_test = train_test_split(DM, Z.ravel(), test_size=0.33) # splits our data
            DM_ols_test[i+1] = DM_test

            # Z_train = mean_scale(Z_train, np.mean(Z_train)) # Mean scaling of the data
            # Z_test = mean_scale(Z_test, np.mean(Z_train)) # Same as above, but should this be done =
            DM_train = mean_scale(DM_train, np.mean(DM_train))

            model_ols[i+1] = OLS(DM_train, Z_train) # Produces betas for our linear regression model

            MSE_ols[i] = MSE(Z_test, DM_test @ model_ols[i+1])
            MSE_ols_train[i] = MSE(Z_train, DM_train @ model_ols[i+1])
            R2_ols[i] = R2(Z_test, DM_test @ model_ols[i+1])
            Var_beta[i+1] = sigma * np.linalg.pinv(DM_test.T @ DM_test).diagonal() # Gives variance in betas (model_ols)

            Var_ols[i] = Var(DM_test @ model_ols[i+1])
            Bias_ols[i] = Bias(Z_test, DM_test @ model_ols[i+1])


        # k = 9
        # plt.errorbar(np.linspace(0,len(model_ols[k]), len(model_ols[k])), model_ols[k], Var_beta[k]) #This will plot betas with variance for polynomianl of degree k
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(1, 2, 1, projection='3d')
        # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # ax.plot_surface(X,Y,Z/np.max(Z))
        # ax2.plot_surface(X,Y, (DM_ols[5]@model_ols[5]).reshape((40,40)) / np.max((DM_ols[5]@model_ols[5])))

        breakpoint()
        exit()

    # Part c
    part_c = False
    if part_c:
        list_of_polydegree = [1, 3, 5, 9, 12, 25]
        # list_of_polydegree = [1, 2, 3, 4, 6, 8]
        boot_ols_mse = np.zeros(len(list_of_polydegree))
        boot_ols_var = np.zeros_like(boot_ols_mse)
        boot_ols_bias = np.zeros_like(boot_ols_mse)
        for i, j in enumerate(list_of_polydegree):
            DM = create_X(X,Y, j)
            DM_train, DM_test, Z_train, Z_test = train_test_split(DM, Z.ravel(), test_size=0.33)
            # Z_train = mean_scale(Z_train, np.mean(Z_train))
            # Z_test = mean_scale(Z_test, np.mean(Z_train))
            boot_ols_mse[i], boot_ols_var[i], boot_ols_bias[i] = bootstrap(DM_train, Z_train, DM_test, Z_test)
        x_ = np.linspace(1,list_of_polydegree[-1],len(list_of_polydegree))
        plt.plot(x_, boot_ols_mse, x_, boot_ols_var, x_, boot_ols_bias)
        plt.legend(["MSE", "Var", "Bias"])
        breakpoint()
