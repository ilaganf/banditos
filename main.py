'''
main.py

Main entrypoint for bandit algorithm on Warfarin dataset
'''
import random

import pandas as pd
import numpy as np

from LinUCB import LinUCB as alg
from ModelBaseClass import ModelBaseClass
from sklearn.linear_model import Ridge

from S1fBaseline import S1fBaseline

# from LassoUCB import LassoUCB as alg
# from FixedDoseBaseline import FixedDoseBaseline as alg
from utils.RefinedDL import RefinedDL as loader

from utils.eval import evaluate
'''
fixed dose: -2319 regret, 53.36% accuracy

age, weight, height, gender, alpha=2.38: -2330 regret, 53.28% accuracy

'''
#FEATURES = ['Weight','indic_male', 'indic_female', 'Age', 'Height']
# #FEATURES = ['Weight', 'indic_male', 'indic_female', 'Age', 'Height', \
#             'indic_*', 'indic_1', 'indic_2', 'indic_3', 'indic_A', 'indic_C',\
#             'indic_G', 'indic_T']
# FEATURES = ['Weight', 'indic_male', 'indic_female', 'Age', 'Height', \
#             'indic_*', 'indic_1', 'indic_2', 'indic_3', 'indic_A', 'indic_C',\
#             'indic_G', 'indic_T', 'Smoker', 'Acetaminophen', 'Asian', 'Black', 'White', 'Race']
# FEATURES = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "med: amiodarone",
#                              "med: carbamazepine", "med: phenytoin", "med: rifampin"]
FEATURES = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "med: amiodarone",\
            "med: carbamazepine", "med: phenytoin", "med: rifampin", "indic_male", "indic_female"]

NUM_TRIALS = 5

def run_s1f():
    #age, weight, height, asian, black or african american,
    #missing or mixed race, Enzyme inducer status, Amiodarone status
    desired_data_features = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "med: amiodarone",
                             "med: carbamazepine", "med: phenytoin", "med: rifampin"]
    # carbamazepine, phenytoin, rifampin, or
    # rifampicin

    data = pd.read_csv('data/warfarin_clean.csv')
    features_of_interest = []
    for name in desired_data_features:
        for feat in data.columns:
            if feat in name:
                features_of_interest.append(feat)

    print("Using {} features".format(len(features_of_interest)))
    print(features_of_interest)

    baseline = S1fBaseline(loader("data/warfarin_clean.csv", features_of_interest, random.randint(1, 100)))

    cum_regret, avg_regret, avg_accuracy = 0, 0, 0
    counts = [0, 0, 0]
    for i in range(NUM_TRIALS):
        reg, avgreg = baseline.evaluate_online()
        avg_accuracy += evaluate(baseline.predictions, baseline.data_loader.labels)
        cum_regret += reg
        avg_regret += avgreg
        for pred in baseline.predictions:
            counts[int(pred)] += 1
        baseline.data_loader.reshuffle()

    cum_regret /= NUM_TRIALS
    avg_regret /= NUM_TRIALS
    avg_accuracy /= NUM_TRIALS
    total = sum(counts)

    print("Results (averaged over {} trials)".format(NUM_TRIALS))
    print("Cumulative Regret {}, Average Regret {}".format(cum_regret, avg_regret))
    print("Accuracy: ", avg_accuracy)
    print("Average low: {} ({}%)".format(counts[0], 100 * (counts[0] / total)))
    print("Average med: {} ({}%)".format(counts[1], 100 * (counts[1] / total)))
    print("Average high: {} ({}%)".format(counts[2], 100 * (counts[2] / total)))

def calc_oracle():
    def convert_labels_to_rewards(labels, action):
        labels = labels.copy()
        base_class = ModelBaseClass(None)
        for i in range(labels.shape[0]):
            labels[i] = 0.0 if base_class.ideal_action(labels[i]) == action else -1.0
        return labels
    def convert_labels_to_actions(labels):
        labels = labels.copy()
        base_class = ModelBaseClass(None)
        for i in range(labels.shape[0]):
            labels[i] = base_class.ideal_action(labels[i])
        return labels

    # data = pd.read_csv('data/warfarin_clean2.csv')
    # features_of_interest = []
    # for name in FEATURES:
    #     for feat in data.columns:
    #         if feat in name: features_of_interest.append(feat)
    features_of_interest = FEATURES
    data_loader = loader("data/warfarin_clean3.csv", features_of_interest, random.randint(1, 100))
    true_actions = convert_labels_to_actions(data_loader.labels.values).copy()
    data = data_loader.data.values.copy()

    Q = np.zeros((3, data.shape[0]))
    for action in range(3):
        labels = convert_labels_to_rewards(data_loader.labels.values, action)
        clf = Ridge(alpha=0.0)
        clf.fit(data, labels)
        Q[action, :] = clf.predict(data)
    pred_actions = np.argmax(Q, axis=0)
    print("% correct is = {}".format(sum(pred_actions == true_actions) / data.shape[0]))


def main():
    data = pd.read_csv('data/warfarin_clean.csv')
    features_of_interest = []
    for feat in data.columns:
        for name in FEATURES:
            if name in feat: features_of_interest.append(feat)

    print("Using {} features".format(len(features_of_interest)))
    print(features_of_interest)

    lin_ucb = alg(loader("data/warfarin_clean.csv", features_of_interest, random.randint(1, 100)))
    
    cum_regret, avg_regret, avg_accuracy = 0, 0, 0
    counts = [0,0,0]
    for i in range(NUM_TRIALS):
        reg, avgreg = lin_ucb.evaluate_online()
        avg_accuracy += evaluate(lin_ucb.predictions, lin_ucb.data_loader.labels)
        cum_regret += reg
        avg_regret += avgreg
        for pred in lin_ucb.predictions:
            counts[int(pred)] += 1
        lin_ucb.data_loader.reshuffle()

    cum_regret /= NUM_TRIALS
    avg_regret /= NUM_TRIALS
    avg_accuracy /= NUM_TRIALS
    total = sum(counts)

    # print(np.sum(lin_ucb.data_loader.labels == 0))
    # print(np.sum(lin_ucb.data_loader.labels == 1))
    # print(np.sum(lin_ucb.data_loader.labels == 2))

    print("Results (averaged over {} trials)".format(NUM_TRIALS))
    # print("Alpha = ", lin_ucb.alpha)
    print("Cumulative Regret {}, Average Regret {}".format(cum_regret, avg_regret))
    print("Accuracy: ", avg_accuracy)
    print("Average low: {} ({}%)".format(counts[0], 100*(counts[0]/total)))
    print("Average med: {} ({}%)".format(counts[1], 100*(counts[1]/total)))
    print("Average high: {} ({}%)".format(counts[2], 100*(counts[2]/total)))

if __name__ == '__main__':
   main()
   #run_s1f()
   #calc_oracle()