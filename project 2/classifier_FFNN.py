import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Layer import Layer
from FFNN import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn import datasets


if __name__ == "__main__":
    data, target = datasets.load_breast_cancer(return_X_y=True)
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)
    SS = StandardScaler()                        # Standard scaling the data to keep the network behaving nicely
    train_data = SS.fit_transform(train_data)
    test_data = SS.fit_transform(test_data)

    if False: # This piece of code will produce the heatmap presented in the result section (FFNN: Classification)
        eta_vals = np.logspace(-6, 0, 7)
        lmbd_vals = np.logspace(-6, 0, 7)
        train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
        test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
        for i, eta in enumerate(eta_vals):
            for j, lmbda in enumerate(lmbd_vals):
                dnn = NeuralNetwork(train_data, train_target,
                        classifier=True,
                        batch_size=50,
                        epochs=50,
                        lmbd=lmbda,
                        eta=eta,
                        n_hidden_layers=2,
                        n_hidden_neurons=10,
                        hidden_act_func="sigmoid")
                dnn.train()
                output, errors = dnn.predict(test_data,test_target)
                train_accuracy[i][j] = errors["train"]
                test_accuracy[i][j] = errors["test"]
        fig, axs = plt.subplots(2, figsize=(8,8))
        sns.heatmap(test_accuracy, annot=True, ax=axs[0], fmt=".2f",
                                xticklabels=lmbd_vals, yticklabels=eta_vals)
        sns.heatmap(train_accuracy, annot=True, ax=axs[1], fmt=".2f",
                                xticklabels=lmbd_vals, yticklabels=eta_vals)
        fig.suptitle("Network with 10 neurons and 2 hidden layers using Sigmoid")
        axs[0].set_title("Test data")
        axs[0].set_xlabel(r"Regularizaton parameter: $\lambda$")
        axs[0].set_ylabel(r"Learning rate: $\eta$")
        axs[1].set_title("Train data")
        axs[1].set_xlabel(r"Regularizaton parameter: $\lambda$")
        axs[1].set_ylabel(r"Learning rate: $\eta$")
        fig.tight_layout()
        plt.savefig("epoch50_bs50_l2_neuron10_sigmoid_heatmap_classifier.pdf")
        plt.show()

    if False: # This piece of code will produce the various tables presented in the section (FFNN: Classification)
        lmbd = 0.001
        eta = 0.001
        epochs = np.array((100,300, 1000))
        batch_size = np.array((20,35,50))
        n_layers = np.array((1,3,10))
        n_neurons = np.array((2,5,10))
        act_func = ["relu","leakyrelu","sigmoid"]
        dnn = NeuralNetwork(train_data, train_target,
                classifier=True,
                batch_size=batch_size[0],
                epochs=500,
                lmbd=lmbd,
                eta=eta,
                n_hidden_layers=n_layers[2],
                n_hidden_neurons=n_neurons[1],
                hidden_act_func=act_func[2])
        dnn.train()
        output, errors = dnn.predict(test_data,test_target)
        print(errors)
