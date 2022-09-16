import numpy as np
from sklearn.model_selection import train_test_split
from random import random, seed

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

def generate_MSE_R2(X,Y,Z):
    """Generates, and calculates Mean Squared Error and R^2 for
        two dimensional polynomials up to a degree of 5.

    -------
    Params:
    - X, Y, Z: NxN matrices of equal size where X,Y are inputs for a function(x,y) that gives Z.


    ----
    Returns:
    MSE, R_squared: Dictionaries containing the values for MSE, R^2 for each of the
                            polynomials up to degree 5
    """
    Z_mean = np.mean(Z.ravel())
    DM = {}
    beta = {}
    Z_tilde = {}
    MSE = {}
    R_squared = {}
    for k in range(1, 6):
        DM[k] = create_X(X.ravel(),Y.ravel(),n=k)
        beta[k] = np.linalg.inv(DM[k].T.dot(DM[k])).dot(DM[k].T).dot(Z.ravel())
        Z_tilde[k] = DM[k].dot(beta[k])
        n = len(Z_tilde[k])
        MSE[k] = 1 / n * sum((Z.ravel()[i] - Z_tilde[k].ravel()[i]) **2 for i in range(len(Z.ravel())))
        R_squared[k] = 1 - ( sum((Z.ravel()[i] - Z_tilde[k].ravel()[i]) **2 for i in range(len(Z.ravel()))) /
                                sum((Z.ravel()[i] - Z_mean) **2 for i in range(len(Z.ravel()))) )

    return MSE, R_squared
if __name__ == "__main__":
    x = np.arange(0,1,0.05)
    y = np.arange(0,1,0.05)
    X,Y = np.meshgrid(x,y)
    Z = FrankeFunction(X,Y)
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X,Y,Z)
    MSE, R_squared = generate_MSE_R2(X_train,Y_train,Z_train)
    breakpoint()
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # breakpoint()
    # surf1 = ax.plot_surface(X,Y,Z)
    # surf2 = ax.plot_surface(X,Y,)
    # plt.show()
