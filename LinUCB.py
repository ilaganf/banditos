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
    
    def __init__(self, data_loader, mlp_dim=None):
        """
        @param data_loader See DataLoader.py
        @param num_features number of features per patient to use for regression
        """
        super(LinUCB, self).__init__(data_loader)
        self.alpha = 1.0 # assuming delta = 0.05 todo: check
        self.num_arms = len(self.actions)
        self.num_features = len(self.data_loader.get_features()) if not mlp_dim else mlp_dim
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
        x_t = patient.T if type(patient) is np.ndarray else patient.values.T # features for the patient, as numpy array of shape (num_features,1)
        expected_rewards = [] # each item is (expected reward, action)
        for action in self.actions:
            theta = np.linalg.solve(self.A[action], self.b[action])
            pred_mean_reward = theta.T @ x_t
            variance_bonus = self.alpha * np.sqrt(x_t.T @ np.linalg.solve(self.A[action], x_t))
            expected_rewards.append((pred_mean_reward + variance_bonus, action))
        next_action = self.select_action(expected_rewards)

        return next_action


    def update_model(self, patient, action, ideal_action, use_modified_reward_extension=False):
        x_t = patient.values.T  # features for the patient, as numpy array of shape (num_features,)
        #x_t = np.reshape(np.append(x_t, [1]), (-1, 1))
        # reward = 0.0 if action == ideal_action else -1.0
        if use_modified_reward_extension:
            if action == ideal_action:
                if action == 1:
                    reward = 1.5
                else:
                    reward = 1
                # if action == 0:
                #     reward = np.random.normal(1.2, 2)
                # else:
                #     reward = np.random.normal(1, 1)
            else:
                reward = -1
            # elif action == 0:
            #     reward = -1
            # elif action == 1:
            #     reward = -1.8
            # else:
            #     reward = -2.0
            # could try +1 for right action as well
        else:
            reward = 0.0 if action == ideal_action else -1.0
        self.A[action] += x_t @ x_t.T
        self.b[action] += reward * x_t

if __name__ == '__main__':
    # lin_ucb = LinUCB(DataLoader("data/warfarin.csv"), 2)
    lin_ucb = LinUCB(RefinedDL("data/warfarin_clean.csv"))
    cum_regret, avg_regret = lin_ucb.evaluate_online()
    print("cum_regret {}, avg_regret {}".format(cum_regret, avg_regret))