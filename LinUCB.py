'''
LinUCB.py

Implements the linear UCB bandit algorithm
'''
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP

from ModelBaseClass import ModelBaseClass
from utils.DataLoader import DataLoader
from utils.RefinedDL import RefinedDL
from sklearn.linear_model import Ridge

class LinUCB(ModelBaseClass):
    
    def __init__(self, data_loader, use_mlp=False, mlp_dim=None):
        """
        @param data_loader See DataLoader.py
        @param num_features number of features per patient to use for regression
        """
        super(LinUCB, self).__init__(data_loader)
        self.alpha = 2  #5.7
        self.num_arms = len(self.actions)
        self.num_features = len(self.data_loader.get_features()) if not mlp_dim else mlp_dim

        self.A = np.zeros((self.num_arms, self.num_features, self.num_features))
        for i in range(self.num_arms):
            self.A[i] = np.eye(self.num_features)
        self.b = np.zeros((self.num_arms, self.num_features, 1))

        self.use_mlp = use_mlp


    def next_action(self, patient, mlp_mode=False):
        """
        @param patient dataframe representing features for the patient
        """
        x_t = patient.T if type(patient) is np.ndarray else patient.values.T # features for the patient, as numpy array of shape (num_features,1)
        expected_rewards = [] # each item is (expected reward, action)
        if not mlp_mode:
            for action in self.actions:
                theta = np.linalg.solve(self.A[action], self.b[action])
                pred_mean_reward = theta.T @ x_t
                variance_bonus = self.alpha * np.sqrt(x_t.T @ np.linalg.solve(self.A[action], x_t))
                expected_rewards.append((pred_mean_reward + variance_bonus, action))
        else:
            for action in self.actions:
                x_a = np.concatenate([x_t, [action + 1]], axis=None)
                expected_rewards.append((self.mlp.predict([x_a])[0], action))

        next_action = self.select_action(expected_rewards)
        return next_action

    def update_model(self, patient, action, ideal_action):
        x_t = patient.T if type(patient) is np.ndarray else patient.values.T  # features for the patient, as numpy array of shape (num_features,)
        reward = 0.0 if action == ideal_action else -1.0
        new_point = np.concatenate([x_t, [action]], axis=None)
        self.data.append(new_point)
        self.labels.append(reward)
        if reward == 0:
            for a in self.actions:
                if a == action: continue
                self.data.append(np.concatenate([x_t, [a]], axis=None))
                self.labels.append(-1.0)
        self.A[action] += x_t @ x_t.T
        self.b[action] += reward * x_t

    def evaluate_online(self, return_regret_list=False):
        """
        Simulates and evaluates online learning model with samples from data_loader
        
        @returns cumulative_regret, avg_regret 
        """
        if not self.use_mlp:
            super(LinUCB, self).evaluate_online()
        else:
            cumulative_regret = 0.0
            switch_index = len(self.data_loader.data) // 2
            patient, ideal_mg_per_week = self.data_loader.sample_next_patient()
            self.predictions = []
            self.data = []
            self.labels = []
            mlp_on = False
            self.mlp = None
            regret_list = []
            while patient is not None:
                if mlp_on and self.mlp is None:
                    self.mlp = MLP(hidden_layer_sizes=(50,50), learning_rate_init=8e-4)
                    self.mlp.fit(self.data, self.labels)
                ideal_action = self.ideal_action(ideal_mg_per_week)
                actual_action = self.next_action(patient, mlp_on)
                if not mlp_on:
                    self.update_model(patient, actual_action, ideal_action)
                if ideal_action != actual_action:
                    cumulative_regret += -1
                regret_list.append(int(ideal_action != actual_action))
                self.predictions.append(actual_action)
                patient, ideal_mg_per_week = self.data_loader.sample_next_patient()
                mlp_on = self.data_loader.ind > switch_index
            if return_regret_list:
                return cumulative_regret, cumulative_regret / self.data_loader.num_samples(), regret_list
            else:
                return cumulative_regret, cumulative_regret / self.data_loader.num_samples()


if __name__ == '__main__':
    # lin_ucb = LinUCB(DataLoader("data/warfarin.csv"), 2)
    lin_ucb = LinUCB(RefinedDL("data/warfarin_clean.csv"))
    cum_regret, avg_regret = lin_ucb.evaluate_online()
    print("cum_regret {}, avg_regret {}".format(cum_regret, avg_regret))