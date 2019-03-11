import ModelBaseClass
import math
import numpy as np


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

        self.A = [np.eye(self.num_arms)] * self.num_arms
        self.b = [np.zeros(self.num_features)] * self.num_arms

    """
    @param patient numpy array of shape (num_features,)
    @param k if not none, use first k features of patient 
    """
    def next_action(self, patient, k=None):
        expected_rewards = [] # each item is (expected reward, action)
        for action in self.actions:
              A_a_inv = np.linalg.inv(self.A[action])
              pred_reward = np.matmul(A_a_inv, self.b[action])
              variance_bonus = self.alpha * np.sqrt(patient.T @ A_a_inv @ patient)
              expected_rewards.append((pred_reward + variance_bonus, action))
        next_action = list(sorted(expected_rewards, key=lambda x: x[0]))[0][1]
        return next_action

    def update_model(self, patient, action, ideal_action):
        reward = 0.0 if action == ideal_action else -1.0
        self.A[action] += patient @ patient.T
        self.b[action] += reward * self.patient


