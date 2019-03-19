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


if __name__ == '__main__':
    lol = MLPBandit('data/warfarin_clean.csv', 2e-4)
    print("Commencing training...")
    lol.train()
    print("Test accuracy: ", lol.test())