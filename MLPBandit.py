'''
MLPBandit.py
'''
from sklearn.neural_network import MLPClassifier as MLP
import pandas as pd
import numpy as np

GROUND = "Therapeutic Dose of Warfarin"

class MLPBandit:
    def __init__(self, data, lr):
        self.mlp = MLP(hidden_layer_sizes=(50,50,50), learning_rate_init=lr, verbose=True)
        raw = pd.read_csv(data).sample(frac=1, replace=False)
        cutoff = int(.8 * len(raw))
        labels = raw[GROUND]
        bins = [0, 3*7,7*7,labels.max()]
        bucketed = pd.cut(labels, bins, include_lowest=True, labels=False)
        self.train_y = bucketed[:cutoff]
        self.test_y = bucketed[cutoff:]
        raw.drop(GROUND, inplace=True, axis=1)
        self.train_x = raw[:cutoff]
        self.test_x = raw[cutoff:]

    def train(self):
        self.mlp.fit(self.train_x, self.train_y)

    def test(self):
        return self.mlp.score(self.test_x, self.test_y)

    def get_activations(self, X):
        hidden_layer_sizes = self.mlp.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + \
            [clf.n_outputs_]
        activations = [X]
        for i in range(clf.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        clf._forward_pass(activations)
        return activations[-2]


if __name__ == '__main__':
    lol = MLPBandit('data/warfarin_clean.csv', 2e-4)
    print("Commencing training...")
    lol.train()
    print("Test accuracy: ", lol.test())