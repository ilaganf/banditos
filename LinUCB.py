'''
LinUCB.py

Implements the linear UCB bandit algorithm
'''
import numpy as np

from ModelBaseClass import ModelBaseClass
from utils.DataLoader import DataLoader
from utils.RefinedDL import RefinedDL
from sklearn.linear_model import Ridge

class LinUCB(ModelBaseClass):
    
    def __init__(self, data_loader):
        """
        @param data_loader See DataLoader.py
        @param num_features number of features per patient to use for regression
        """
        super(LinUCB, self).__init__(data_loader)
        self.alpha = 2.1 # assuming delta = 0.05 todo: check
        self.num_arms = len(self.actions)
        self.num_features = len(self.data_loader.get_features())
        self.data = []
        self.labels = []

        self.A = np.zeros((self.num_arms, self.num_features, self.num_features))
        for i in range(self.num_arms):
            self.A[i] = np.eye(self.num_features)
        self.b = np.zeros((self.num_arms, self.num_features, 1))

    
    def next_action(self, patient):
        """
        @param patient dataframe representing features for the patient
        """
        x_t = patient.values.T # features for the patient, as numpy array of shape (num_features,)
        expected_rewards = [] # each item is (expected reward, action)
        for action in self.actions:
            A_a_inv = np.linalg.inv(self.A[action])
            pred_reward = x_t.T @ A_a_inv @ self.b[action]
            variance_bonus = self.alpha * np.sqrt(x_t.T @ A_a_inv @ x_t)
            # variance_bonus = self.alpha * np.sqrt(np.log(sum(self.times_action_taken)) / (2.0 * self.times_action_taken[action]))
            expected_rewards.append((pred_reward + variance_bonus, action))
        next_action = self.select_action(expected_rewards)

        return next_action


    def update_model(self, patient, action, ideal_action):
        x_t = patient.values.T[:self.num_features]  # features for the patient, as numpy array of shape (num_features,)
        reward = 0.0 if action == ideal_action else -1.0
        self.A[action] += x_t @ x_t.T
        self.b[action] += reward * x_t

if __name__ == '__main__':
    # lin_ucb = LinUCB(DataLoader("data/warfarin.csv"), 2)
    lin_ucb = LinUCB(RefinedDL("data/warfarin_clean.csv"))
    cum_regret, avg_regret = lin_ucb.evaluate_online()
    print("cum_regret {}, avg_regret {}".format(cum_regret, avg_regret))