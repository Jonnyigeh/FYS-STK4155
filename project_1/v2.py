import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from random import random, seed
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sys import exit

seed(2001)

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

def LASSO(DM, Z, lmb):
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
    r2 = 1 - np.sum((Z.ravel() - Z_model) ** 2) / np.sum((Z.ravel() - Z_mean) **2)
    return r2

def Var(model):
    variance = np.mean( np.var(model) )

    return variance

def Bias(data, model):
    bias = np.mean( (data.ravel() - np.mean(model)) **2)

    return bias

def mean_scale(data, mean_data=0):
    """Mean-scales the dataset by the mean of each column vector in data

    args:
        data        (np.array): meshgrid of datapoints
        mean_data   (np.array): meshgrid of mean of datapoints (mean of columns of data)

    returns:
        new_data    (np.array): Meshgrid of scaled datapoints (now also centered)
    """
    if mean_data==0:
        mean_data = np.mean(data, axis=0, keepdims=True)

    new_data = ( data - mean_data )

    return new_data, mean_data

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

def bootstrap(X, Y, method="OLS", lmbda=0, niteration=100):
    """ Performs bootstrap
    args:

    """
    X_train, X_test, Y_train, Y_test = train_test_split(DM, Z.ravel(), test_size=0.33)

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
        X_train, XmeanT = mean_scale(X_train)
        Y_train, YmeanT = mean_scale(Y_train)

        X_test = X_test - XmeanT
        Y_test = Y_test - YmeanT
        for i in range(niteration):
            new_X, new_Y = resample(X_train, Y_train)
            model[:,i] = X_test @ Ridge(new_X, new_Y, lmbda).ravel() + YmeanT

    if method=="Lasso":
        X_train, XmeanT = mean_scale(X_train)
        Y_train, YmeanT = mean_scale(Y_train)

        X_test = X_test - XmeanT
        Y_test = Y_test - YmeanT
        for i in range(niteration):
            new_X, new_Y = resample(X_train, Y_train)
            model[:,i] = X_test @ LASSO(new_X, new_Y, lmbda).ravel()  + YmeanT


    Y_test = Y_test.reshape(np.shape(Y_test)[0],1)
    error = np.mean( np.mean((Y_test - model)**2, axis=1, keepdims=True) )
    bias = np.mean( (Y_test - np.mean(model, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(model, axis=1, keepdims=True) )

    return error, variance, bias



def cross_val(X, Y, method="OLS", lmbda=0, k=10, return_beta=False):
    kfold = KFold(n_splits=k)
    score_KFold = np.zeros(k)

    if method=="OLS":
        for i, (train_inds, test_inds) in enumerate(kfold.split(X)):
            X_train = X[train_inds]
            Y_train = Y[train_inds]

            X_test = X[test_inds]
            Y_test = Y[test_inds]

            # X_train, XmeanT = mean_scale(X_train)
            # Y_train, YmeanT = mean_scale(Y_train)
            # X_test = X_test - XmeanT
            # Y_test = Y_test - YmeanT
            beta = OLS(X_train, Y_train)
            model = X_test @ beta
            score_KFold[i] = MSE(Y_test, model)

    if method=="Ridge":
        for i, (train_inds, test_inds) in enumerate(kfold.split(X)):
            X_train = X[train_inds]
            Y_train = Y[train_inds]

            X_test = X[test_inds]
            Y_test = Y[test_inds]

            X_train, XmeanT = mean_scale(X_train)
            Y_train, YmeanT = mean_scale(Y_train)
            X_test = X_test - XmeanT
            Y_test = Y_test - YmeanT

            beta = Ridge(X_train, Y_train, lmbda)
            model = X_test @ beta + YmeanT
            score_KFold[i] = MSE(Y_test, model)

    if method=="Lasso":
        for i, (train_inds, test_inds) in enumerate(kfold.split(X)):
            X_train = X[train_inds]
            Y_train = Y[train_inds]

            X_test = X[test_inds]
            Y_test = Y[test_inds]

            X_train, XmeanT = mean_scale(X_train)
            Y_train, YmeanT = mean_scale(Y_train)
            X_test = X_test - XmeanT
            Y_test = Y_test - YmeanT

            beta = LASSO(X_train, Y_train , lmbda)
            model = X_test @ beta + YmeanT
            score_KFold[i] = MSE(Y_test, model)

    estimated_score_KFold = np.mean(score_KFold)
    if return_beta:                         # If one would like to plot the model
        return estimated_score_KFold, beta

    return estimated_score_KFold



if __name__ == "__main__":
    Frankedata = False
    if Frankedata:       # Setting up datastructures
        n = 30         # Number of datapoints
        nlambdas = 10  # Number of different lambdas
        x = np.linspace(0,1,n)
        y = np.linspace(0,1,n)
        X,Y = np.meshgrid(x,y)
        lmbdas = np.logspace(-4, 1, nlambdas)
        sigma = 0.15
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
                boot_ols_mse[i], boot_ols_var[i], boot_ols_bias[i] = bootstrap_OLS(DM_, Z_onedim, niteration=100)

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
            MSE_ols_crossval = np.zeros_like(R2_ols)
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

        # Part c and d
        part_c = False
        if part_c:
            # list_of_polydegree = [1, 3, 5, 9, 12, 25]
            list_of_polydegree = [1, 2, 3, 4, 5]
            boot_ols_mse = np.zeros(len(list_of_polydegree))
            boot_ols_var = np.zeros_like(boot_ols_mse)
            boot_ols_bias = np.zeros_like(boot_ols_mse)
            MSE_ols_crossval = np.zeros_like(boot_ols_mse)
            MSE_ols_crossval2 = np.zeros_like(boot_ols_mse)
            MSE_ols_crossval3 = np.zeros_like(boot_ols_mse)

            for i, j in enumerate(list_of_polydegree):
                DM = create_X(X,Y, j)
                MSE_ols_crossval2[i] = cross_val(DM, Z.ravel(), k = 10)
                MSE_ols_crossval[i] = cross_val(DM, Z.ravel(), k = 5)
                MSE_ols_crossval3[i] = cross_val(DM, Z.ravel(), k = 15)
                boot_ols_mse[i], boot_ols_var[i], boot_ols_bias[i] = bootstrap(DM, Z, niteration=5)
            x_ = np.linspace(1,list_of_polydegree[-1],len(list_of_polydegree))
            # plt.plot(x_, boot_ols_mse, x_, boot_ols_var, x_, boot_ols_bias)
            # plt.legend(["MSE", "Var", "Bias"])

            plt.plot(list_of_polydegree, boot_ols_mse, "-o", list_of_polydegree, MSE_ols_crossval, "-o",
                    list_of_polydegree, MSE_ols_crossval2, "-o", list_of_polydegree, MSE_ols_crossval3, "-o")
            plt.legend(["Bootstrap", "Crossval k = 5", "Crossval k = 10", "Crossval k = 15"])


            breakpoint()
            exit()


        ridge = False
        if ridge:
            lop = [1,2,3,4,5,10]
            lop2 = [1,3,5,10,12,21]
            MSE_cv = np.zeros((nlambdas,len(lop2)))
            MSE_cv2 = np.zeros_like(MSE_cv)
            MSE_cv3 = np.zeros_like(MSE_cv)
            MSE_bs = np.zeros_like(MSE_cv)
            var_bs = np.zeros_like(MSE_cv)
            bias_bs = np.zeros_like(MSE_cv)
            for k, lmb in enumerate(lmbdas):
                for i, j in enumerate(lop2):
                    DM = create_X(X,Y,j)
                    MSE_cv[k,i] = cross_val(DM, Z.ravel(), k=5, method="Ridge", lmbda=lmb)
                    MSE_cv2[k,i] = cross_val(DM, Z.ravel(), k=10, method="Ridge", lmbda=lmb)
                    MSE_cv3[k,i] = cross_val(DM, Z.ravel(), k=15, method="Ridge", lmbda=lmb)

                    MSE_bs[k,i], var_bs[k,i], bias_bs[k,i] = bootstrap(DM, Z,
                                            method="Ridge", lmbda=lmb, niteration=15)

            deg_indx = 1
            plt.plot(lop2, MSE_bs[deg_indx],"-o",lop2, var_bs[deg_indx],"-o",lop2, bias_bs[deg_indx], "-o")
            plt.legend(["MSE", "Var", "Bias"])
            breakpoint()
            exit()



        lasso = False
        if lasso:
            import warnings
            warnings.filterwarnings('ignore') # To combat spam from  sklearn saying our model does not converge
            lop  = [1,2,3,4,5,10]
            lop2 = [1,3,5,10,12,21]
            MSE_cv = np.zeros((nlambdas,len(lop2)))
            MSE_cv2 = np.zeros_like(MSE_cv)
            MSE_cv3 = np.zeros_like(MSE_cv)
            MSE_bs = np.zeros_like(MSE_cv)
            var_bs = np.zeros_like(MSE_cv)
            bias_bs = np.zeros_like(MSE_cv)
            for k, lmb in enumerate(lmbdas):
                for i, j in enumerate(lop2):
                    DM = create_X(X,Y,j)
                    MSE_cv[k,i] = cross_val(DM, Z.ravel(), k=5, method="Lasso", lmbda=lmb)
                    MSE_cv2[k,i] = cross_val(DM, Z.ravel(), k=10, method="Lasso", lmbda=lmb)
                    MSE_cv3[k,i] = cross_val(DM, Z.ravel(), k=15, method="Lasso", lmbda=lmb)

                    MSE_bs[k,i], var_bs[k,i], bias_bs[k,i] = bootstrap(DM, Z,
                                            method="Lasso", lmbda=lmb, niteration=15)
            deg = 1
            # plt.plot(np.log10(lmbdas), MSE_cv[:,deg],"-o", np.log10(lmbdas), MSE_cv2[:,deg],"-o", np.log10(lmbdas),
            #                         MSE_cv3[:,deg],"-o", np.log10(lmbdas), MSE_bs[:,deg], "--o")
            # plt.legend(["Crossval: k=5", "Crossval: k=10", "Crossval: k=15", "Bootstrap"])
            plt.plot(lop2, MSE_bs[deg],"-o",lop2, var_bs[deg],"-o",lop2, bias_bs[deg], "-o")
            plt.title(rf"$\lambda = {lmbdas[deg]:.6f}$")
            plt.ylabel("Error estimate")
            plt.xlabel("Model complexity (poly degree)")
            plt.legend(["MSE", "Var", "Bias"])


            breakpoint()
            exit()


    realdata = True
    if realdata:
        import warnings
        warnings.filterwarnings('ignore') # To combat spam from sklearn.lasso saying our model does not converge
        # Load the terrain
        terrain = imread("SRTM_data_Norway_1.tif")
        # terrain = terrain1[::25,::25]

        nlambdas = 10
        deg = 5                 # polynomial degree
        x = np.linspace(0,1,len(terrain[0,:]))
        y = np.linspace(0,1,len(terrain[:,0]))
        X,Y = np.meshgrid(x,y)
        lmbdas = np.logspace(-4,1, nlambdas)
        Z = terrain

        DM = create_X(X,Y,deg)
        """ Pre-processing of data """ # We are only using CV so this is not needed
        # DM_train, DM_test, Z_train, Z_test = train_test_split(DM, Z.ravel(), test_size=0.33)
        # DM_train, mean_dmtrain = mean_scale(DM_train)
        # Z_train, mean_ztrain = mean_scale(Z_train)
        # DM_test = DM_test - mean_dmtrain
        # Z_test = Z_test - mean_ztrain

        """ OLS """
        MSE_cv_ols = np.zeros(nlambdas)
        ols_beta = np.zeros((nlambdas, len(DM[0,:])))

        """ Ridge """
        MSE_cv_ridge = np.zeros(nlambdas)
        ridge_beta = np.zeros_like(ols_beta)

        """ Lasso """
        MSE_cv_lasso = np.zeros(nlambdas)
        lasso_beta = np.zeros_like(ols_beta)

        for i, lmb in enumerate(lmbdas):
            MSE_cv_ols[i], ols_beta[i,:] = cross_val(DM, Z.ravel(), k=10, method="OLS", lmbda=lmb, return_beta=True)
            MSE_cv_ridge[i], ridge_beta[i,:] = cross_val(DM, Z.ravel(), k=10, method="Ridge", lmbda=lmb, return_beta=True)
            MSE_cv_lasso[i], lasso_beta[i,:] = cross_val(DM, Z.ravel(), k=10, method="Lasso", lmbda=lmb, return_beta=True)

        breakpoint()


        # MSE_bs_ols = np.zeros(nlambdas)
        # var_bs = np.zeros(nlambdas)
        # bias_bs = np.zeros(nlambdas)
        # MSE_bs_lasso = np.zeros(nlambdas)
        # var_bs
        # bias_bs
        # MSE_bs_ridge = np.zeros(nlambdas)
        # var_bs = np.zeros(nlambdas)
        # bias_bs = np.zeros(nlambdas)
        #
        # # Show the terrain
        # plt.figure()
        # plt.title("Terrain over Norway 1")
        # plt.imshow(terrain1, cmap="gray")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.show()
