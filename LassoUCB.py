'''
LassoUCB.py
'''

import numpy as np
# from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from ModelBaseClass import ModelBaseClass
from utils.RefinedDL import RefinedDL

class LassoUCB(ModelBaseClass):

    def __init__(self, data_loader):
        super(LassoUCB, self).__init__(data_loader)
        self.alpha1 = 1.2
        self.alpha2 = 2.0
        self.num_arms = len(self.actions)
        self.lassos = [Ridge(self.alpha1) for _ in range(self.num_arms)]
        dummy = np.zeros((1, len(self.data_loader.get_features())))
        for i in range(self.num_arms):
            self.lassos[i].fit(dummy, [-1])

    def next_action(self, patient):
        """
        @param patient dataframe representing features for the patient
        """
        x = patient.values # features for the patient, as numpy array of shape (1,num_features)
        expected_rewards = [] # each item is (expected reward, action)
        for action in self.actions:
            pred = self.lassos[action].predict(x)
            bonus = self.alpha2 * np.sqrt(np.log(sum(self.times_action_taken)) / \
                    (2.0 * self.times_action_taken[action]))
            expected_rewards.append((pred+bonus, action))
        next_action = max(expected_rewards)[1]
        return next_action


    def update_model(self, patient, action, ideal_action):
        x_t = np.squeeze(patient.values)
        reward = 0.0 if action == ideal_action else -1.0
        self.seen_data[action].append((x_t, reward))
        X, y = [], []
        for data, r in self.seen_data[action]:
            X.append(data)
            y.append(r)
        self.lassos[action].fit(X, y)