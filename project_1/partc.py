import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import random, seed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def create_X(x,y,n):
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

def generate_linear_model(DM, Z):
    """

    """
    beta = {}
    for k in range(1, 6):
        beta[k] = np.linalg.inv(DM[k].T.dot(DM[k])).dot(DM[k].T).dot(Z.ravel())

    return beta

def generate_DM(X,Y):
    """

    """
    DM = {}
    for k in range(1,6):
        DM[k] = create_X(X.ravel(),Y.ravel(),n=k)

    return DM

def generate_MSE_R2(DM, beta, Z):
    """


    """
    Z_mean = np.mean(Z.ravel())
    Z_tilde = {}
    MSE = {}
    R_squared = {}
    for k in range(1, 6):
        Z_tilde[k] = DM[k].dot(beta[k])
        n = len(Z_tilde[k])
        MSE[k] = 1 / n * sum((Z.ravel()[i] - Z_tilde[k].ravel()[i]) **2 for i in range(len(Z.ravel())))
        R_squared[k] = 1 - ( sum((Z.ravel()[i] - Z_tilde[k].ravel()[i]) **2 for i in range(len(Z.ravel()))) /
                                sum((Z.ravel()[i] - Z_mean) **2 for i in range(len(Z.ravel()))) )

    return MSE, R_squared


# def Bootstrap(Bias,Var,err):

#     "





#     "
#     return ..






if __name__ == "__main__":
    bootstrap_n = 100
    x = np.arange(0,1,0.05)
    y = np.arange(0,1,0.05)
    X,Y = np.meshgrid(x,y)
    xnoise = np.random.rand(len(x)) * 0.25
    ynoise = np.random.rand(len(x)) * 0.25
    Xnoise, Ynoise = np.meshgrid(xnoise, ynoise)
    Z = FrankeFunction(X,Y) + (Xnoise + Ynoise)
    DM = generate_DM(X,Y)
    DM_train = {}
    DM_test = {}



    for k in range(1,6):
        DM_train[k], DM_test[k], Z_train, Z_test = train_test_split(DM[k], Z.ravel(), test_size = 0.33)
    
    
    
    for i in range(bootstrap_n):
        x_, y_ = resample(DM_train[3],Z_train)
        Z_pred[:, i] = model.fit(DM_train[3],Z_train).predict(DM_test[3]).ravel()
    bias = np.mean( (Z_test - Z_pred) **2, axis = 1, keepdims = True)
    variance = np.mean( np.var(Z_pred, axis = 1 , keepdim = True) )


    beta = generate_linear_model(DM_train, Z_train)
    MSE_2, R2_2 = generate_MSE_R2(DM_train, beta, Z_train)
    MSE, R2 = generate_MSE_R2(DM_test, beta, Z_test)
    x = np.linspace(1,21,21)
    fig, axs = plt.subplots(2, 2)
    b = np.linspace(0,21,21)
    print('Bias$^2 :$', bias )
    print('Variance :', variance )

    # axs[0][0].plot([1,2,3,4,5], [MSE[1], MSE[2], MSE[3], MSE[4], MSE[5]], "-o")
    # axs[0][0].plot([1,2,3,4,5], [R2[1], R2[2], R2[3], R2[4], R2[5]], "-o")
    # axs[0][0].legend(["MSE", "R2"])
    # axs[0][1].scatter(x[0:3], beta[1])
    # axs[1][0].scatter(x[0:10], beta[3])
    # axs[1][1].scatter(x, beta[5])
    # axs[0][1].legend(["Beta for polynomial degree 1"])
    # axs[1][0].legend(["Beta for polynomial degree 3"])
    # axs[1][1].legend(["Beta for polynomial degree 5"])

    plt.show()
    breakpoint()