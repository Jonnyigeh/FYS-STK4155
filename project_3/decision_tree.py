import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
np.random.seed(100)

def bias_var_tradeoff(
            X_train,
            Y_train,
            X_test,
            Y_test,
            max_depth=15,
            n_iterations=100,
            ):
    biases = np.zeros(max_depth-2)
    variances = np.zeros_like(biases)
    mse = np.zeros_like(biases)

    for i in range(2, max_depth):
        predictions = np.zeros((Y_test.shape[0], n_iterations))
        for j in range(n_iterations):   # Bootstrap
            tree = DecisionTreeRegressor(max_depth=i)
            # Resample the training data
            new_X, new_Y = resample(X_train, Y_train)
            # Train the decision tree on the training data
            tree.fit(new_X, new_Y)
            tree_pred = tree.predict(X_test)
            predictions[:,j] = tree_pred

        # Computing the various error estimates for the bias-variance tradeoff
        mse[i-2] = np.mean( np.mean(np.square(Y_test - predictions), axis=1, keepdims=True) )
        biases[i-2] = np.mean( np.square(Y_test - np.mean(predictions, axis=1, keepdims=True)) )
        variances[i-2] = np.mean( np.var(predictions, axis=1, keepdims=True) )

    return biases, variances, mse

if __name__ == "__main__":
    sns.set_theme()
    # Producing some dataset, and making some test data to evaluate our tree
    x = np.linspace(-2, 2, 100)
    x = x.reshape(len(x),1)
    f = lambda x: np.sin(0.4 * x) + np.cos(3 * x) + np.log10(0.1 + abs(x))
    y = f(x) + 0.15 * np.random.randn(*x.shape)
    X_train, Y_train = x, y
    X_test = np.linspace(-2, 2, int(len(x)/4)).reshape(int(len(x)/4),1)
    Y_test = f(X_test)

    if True:
        X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2)
        max_depth = 12
        b, v, m = bias_var_tradeoff(X_train, Y_train, X_test, Y_test, max_depth=max_depth)
        n_depths = np.linspace(2,max_depth,len(b))
        plt.plot(n_depths, b, n_depths, v, n_depths, m)
        plt.legend(["bias", "variance", "mse"])
        plt.title("Bias-Variance tradeoff for Decision Tree Regressor for rising complexity")
        plt.xlabel("Complexity of model: Max depth of tree")
        plt.ylabel("Error estimate")
        # plt.savefig("BVtrade_dec_tree.pdf")
        plt.show()

    if False:

        # Create the decision trees
        tree_1 = DecisionTreeRegressor(max_depth=2)
        tree_2 = DecisionTreeRegressor(max_depth=5)
        tree_3 = DecisionTreeRegressor(max_depth=7)

        # Train the decision tree on the training data
        tree_1.fit(X_train, Y_train)
        tree_2.fit(X_train, Y_train)
        tree_3.fit(X_train, Y_train)

        # Use the trained decision tree to make predictions on the test data
        tree_pred1 = tree_1.predict(X_test)
        tree_pred2 = tree_2.predict(X_test)
        tree_pred3 = tree_3.predict(X_test)

        # Visualization of the predicted data
        plt.plot(x[::40],y[::40], "ko", label="data")
        plt.plot(X_test, tree_pred1, "r-", label="max_depth = 2")
        plt.plot(X_test, tree_pred2, "g-", label="max_depth = 5")
        plt.plot(X_test, tree_pred3, "b-", label="max_depth = 7")
        plt.title("DT from Scikit learn: Visualization of the model data")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.legend()
        # plt.savefig("DT_visualization.pdf")
        plt.show()
