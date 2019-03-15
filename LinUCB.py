from ModelBaseClass import ModelBaseClass
import math
import numpy as np
from utils.DataLoader import DataLoader

class LinUCB(ModelBaseClass):
    """
    @param data_loader See DataLoader.py
    @param num_features number of features per patient to use for regression
    """
    def __init__(self, data_loader, num_features):
        super(LinUCB, self).__init__(data_loader)
        self.alpha = 2.36 # assuming delta = 0.05 todo: check
        self.num_arms = len(self.actions)
        self.num_features = num_features

        self.A = np.zeros((self.num_arms, self.num_features, self.num_features))
        for i in range(self.num_arms):
            self.A[i] = np.eye(self.num_features)
        self.b = np.zeros((self.num_arms, self.num_features))
        # self.A = [np.eye(self.num_features) for _ in range(self.num_arms) ]
        # self.b = [np.zeros(self.num_features) for _ in range(self.num_arms) ]

    """
    @param patient dataframe representing features for the patient
    """
    # TODO: we need to sanitize the input to get rid of the NaN's
    def next_action(self, patient):
        x_t = patient.values.T[:self.num_features] # features for the patient, as numpy array of shape (num_features,)
        expected_rewards = [] # each item is (expected reward, action)
        for action in self.actions:
              A_a_inv = np.linalg.inv(self.A[action])
              pred_reward = x_t.T @ A_a_inv @ self.b[action]
              variance_bonus = self.alpha * np.sqrt(x_t.T @ A_a_inv @ x_t)
              expected_rewards.append((np.asscalar(pred_reward[0] + variance_bonus), action))
        next_action = list(sorted(expected_rewards, key=lambda x: x[0]))[-1][1]
        return next_action

    def update_model(self, patient, action, ideal_action):
        x_t = patient.values.T[:self.num_features]  # features for the patient, as numpy array of shape (num_features,)
        reward = 0.0 if action == ideal_action else -1.0
        self.A[action] += x_t @ x_t.T
        self.b[action] += reward * np.squeeze(x_t, -1)


lin_ucb = LinUCB(DataLoader("data/warfarin.csv"), 1)
cum_regret, avg_regret = lin_ucb.evaluate_online()
print("cum_regret {}, avg_regret {}".format(cum_regret, avg_regret))