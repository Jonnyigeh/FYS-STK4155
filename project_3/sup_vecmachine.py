import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
np.random.seed(0)

def bias_var_tradeoff(
            X_train,
            Y_train,
            X_test,
            Y_test,
            c=1,
            n_iterations=10,
            ):

    predictions = np.zeros((Y_test.shape[0], n_iterations))
    model = SVR(kernel="rbf", C=c)
    for j in range(n_iterations):   # Bootstrap
        # Resample the training data
        new_X, new_Y = resample(X_train, Y_train.ravel())
        # Train the SVM on the new training data
        model.fit(new_X, new_Y)
        model_pred = model.predict(X_test)
        predictions[:,j] = model_pred

    # Computing the various error estimates for the bias-variance tradeoff
    mse = np.mean( np.mean(np.square(Y_test - predictions), axis=1, keepdims=True) )
    bias = np.mean( np.square(Y_test - np.mean(predictions, axis=1, keepdims=True)) )
    variance = np.mean( np.var(predictions, axis=1, keepdims=True) )

    return bias, variance, mse


if __name__ == "__main__":
    sns.set_theme()
    # Producing some dataset, and making some test data to evaluate our tree
    x = np.linspace(-2, 2, 100)
    x = x.reshape(len(x),1)
    f = lambda x: np.sin(0.4 * x) + np.cos(3 * x) + np.log10(0.1 + abs(x))
    y = f(x) + 0.15 * np.random.randn(*x.shape)
    X_train, Y_train = x, y

    if False:
        X_test = np.linspace(-2, 2, int(len(x)/4)).reshape(int(len(x)/4),1)
        Y_test = f(X_test)
        # Create the SVM models
        model1 = SVR(kernel="rbf",C=0.1)
        model2 = SVR(kernel="rbf",C=1)
        model3 = SVR(kernel="rbf",C=0.01)

        # Fit the model on our training data
        model1.fit(X_train, Y_train.ravel())
        model2.fit(X_train, Y_train.ravel())
        model3.fit(X_train, Y_train.ravel())

        # Making predictions on the test data
        model_pred1 = model1.predict(X_test)
        model_pred2 = model2.predict(X_test)
        model_pred3 = model3.predict(X_test)
        # Visualization of the predicted data
        plt.plot(x,y, "ko", label="data")
        plt.plot(X_test, model_pred1, "r-", label="C = 0.1")
        plt.plot(X_test, model_pred2, "b-", label="C = 1")
        plt.plot(X_test, model_pred3, "g-", label="C = 0.01")
        plt.title("SVR from Scikit learn: Visualization of the model data")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.legend()
        # plt.savefig("SVR_visualization.pdf")
        plt.show()

    if True:
        X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3)
        reg_params = np.array([0.0001, 0.001, 0.01, 0.1, 1, 1.2])
        b = np.zeros(len(reg_params))
        v = np.zeros_like(b)
        m = np.zeros_like(b)
        for i, c in enumerate(reg_params):
            b[i], v[i], m[i] = bias_var_tradeoff(X_train, Y_train, X_test, Y_test, c=c)

        x_axis = np.linspace(0.0001, 1.2, len(b))
        plt.plot(x_axis, b, x_axis, v, x_axis, m)
        plt.title("Bias-Variance tradeoff for Support Vector Regressor for rising complexity")
        plt.legend(["bias", "variance", "mse"])
        plt.xlabel("Complexity of model: Larger values of reg.param")
        plt.ylabel("Error estimate")
        # plt.savefig("BVtrade_svr.pdf")
        plt.show()
