'''
MLPBandit.py
'''
from sklearn.neural_network import MLPClassifier as MLP
import pandas as pd
import numpy as np

GROUND = 'Therapeutic Dose of Warfarin'

class MLPBandit:
    def __init__(self, data, lr, final_hidden_dim=50):
        self.mlp = MLP(hidden_layer_sizes=(50,50,final_hidden_dim), learning_rate_init=lr, verbose=True)
        if type(data) is str:
            raw = pd.read_csv(data).sample(frac=1, replace=False)
        else:
            raw = data
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
            [self.mlp.n_outputs_]
        activations = [X]
        for i in range(self.mlp.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        self.mlp._forward_pass(activations)
        return activations[-2]


if __name__ == '__main__':
    data = pd.read_csv('data/warfarin_clean.csv').sample(frac=1, random_state=42)
    cutoff = int(.15 * len(data))
    test_set = data[:cutoff]
    train_set = data[cutoff:]
    best_model = None
    best_val = 0
    for x in range(10):
        print("\nValidation %d start"%(x+1))
        lol = MLPBandit(train_set.copy(), 4e-4)
        lol.train()
        val = lol.test()
        if val > best_val:
            best_val = val
            best_model = lol
        print("Iteration {} validation accuracy: {}\n".format(x+1,lol.test()))
    labels = test_set['Therapeutic Dose of Warfarin']
    bins = [0, 3*7,7*7,labels.max()]
    test_labels = pd.cut(labels, bins, include_lowest=True, labels=False)
    test_x = test_set.drop('Therapeutic Dose of Warfarin', axis=1, inplace=True)
    print("Test accuracy: ", best_model.mlp.score(test_x, test_labels))