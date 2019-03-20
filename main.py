'''
main.py

Main entrypoint for bandit algorithm on Warfarin dataset
'''
import random

import pandas as pd
import numpy as np

from LinUCB import LinUCB as alg
from ModelBaseClass import ModelBaseClass
from MLPBandit import MLPBandit as MLPBandit
from sklearn.linear_model import Ridge
from SimpleLinearAlg import SimpleLinearAlg
import matplotlib.pyplot as plt
from FixedDoseBaseline import FixedDoseBaseline

from S1fBaseline import S1fBaseline

# from LassoUCB import LassoUCB as alg
from FixedDoseBaseline import FixedDoseBaseline as fixedalg
from utils.RefinedDL import RefinedDL as loader
import scipy

from utils.eval import evaluate
'''
fixed dose: -2319 regret, 53.36% accuracy

age, weight, height, gender, alpha=2.38: -2330 regret, 53.28% accuracy

'''
#FEATURES = ['Weight','indic_male', 'indic_female', 'Age', 'Height']
# #FEATURES = ['Weight', 'indic_male', 'indic_female', 'Age', 'Height', \
#             'indic_*', 'indic_1', 'indic_2', 'indic_3', 'indic_A', 'indic_C',\
#             'indic_G', 'indic_T']
FEATURES = ['Weight', 'indic_male', 'indic_female', 'Age', 'Height', \
            'indic_*', 'indic_1', 'indic_2', 'indic_3', 'indic_A', 'indic_C',\
            'indic_G', 'indic_T', 'Smoker', 'Acetaminophen', 'Asian', 'Black', 'White', 'Race']
# FEATURES = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "med: amiodarone",
#                              "med: carbamazepine", "med: phenytoin", "med: rifampin"]

#FEATURES = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "Mixed race"]
FEATURES = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American",
                     "Unknown or mixed race", "Amiodarone (Cordarone)", "Enzyme inducer status"]

# FEATURES = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "med: amiodarone",\
#             "med: carbamazepine", "med: phenytoin", "med: rifampin", "indic_male", "indic_female"]

NUM_TRIALS = 1

def run_s1f():
    #age, weight, height, asian, black or african american,
    #missing or mixed race, Enzyme inducer status, Amiodarone status
    # desired_data_features = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "med: amiodarone",
    #                          "med: carbamazepine", "med: phenytoin", "med: rifampin"]
    # carbamazepine, phenytoin, rifampin, or
    # rifampicin
    desired_data_features = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American",
                     "Unknown or mixed race", "Amiodarone (Cordarone)", "Enzyme inducer status"]

    data = pd.read_csv('data/warfarin_clean5.csv')
    features_of_interest = []
    for name in desired_data_features:
        for feat in data.columns:
            if feat in name:
                features_of_interest.append(feat)

    print("Using {} features".format(len(features_of_interest)))
    print(features_of_interest)

    baseline = S1fBaseline(loader("data/warfarin_clean5.csv", features_of_interest))

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
    data_loader = loader("data/warfarin_clean7.csv", features_of_interest)
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

def mlp_test():
    data = 'data/warfarin_clean.csv'
    mlp = MLPBandit(data, 5e-4)
    load = loader('data/warfarin_clean.csv', mlp=mlp)
    bandit = alg(load, mlp_dim=50)
    counts = [0,0,0]
    cum_regret, avg_regret, avg_accuracy = 0, 0, 0
    for i in range(NUM_TRIALS):
        reg, avgreg = bandit.evaluate_online()
        avg_accuracy += evaluate(bandit.predictions, bandit.data_loader.labels)
        cum_regret += reg
        avg_regret += avgreg
        for pred in bandit.predictions:
            counts[int(pred)] += 1
        bandit.data_loader.reshuffle()

    cum_regret /= NUM_TRIALS
    avg_regret /= NUM_TRIALS
    avg_accuracy /= NUM_TRIALS
    total = sum(counts)

    print("Cumulative Regret {}, Average Regret {}".format(cum_regret, avg_regret))
    print("Accuracy: ", avg_accuracy)
    print("Average low: {} ({}%)".format(counts[0], 100*(counts[0]/total)))
    print("Average med: {} ({}%)".format(counts[1], 100*(counts[1]/total)))
    print("Average high: {} ({}%)".format(counts[2], 100*(counts[2]/total)))


def run_modified_ucb():
    data = pd.read_csv('data/warfarin_clean6.csv')
    features_of_interest = []
    for feat in data.columns:
        for name in FEATURES:
            if name in feat: features_of_interest.append(feat)

    print("Using {} features".format(len(features_of_interest)))
    print(features_of_interest)

    modified_ucb = SimpleLinearAlg(loader("data/warfarin_clean6.csv", features=features_of_interest, seed=random.randint(1, 100)))

    cum_regret, avg_regret, avg_accuracy = 0, 0, 0
    counts = [0, 0, 0]
    NUM_TRIALS = 1
    for i in range(NUM_TRIALS):
        reg, avgreg = modified_ucb.evaluate_online()
        avg_accuracy += evaluate(modified_ucb.predictions, modified_ucb.data_loader.labels)
        cum_regret += reg
        avg_regret += avgreg
        for pred in modified_ucb.predictions:
            counts[int(pred)] += 1
        modified_ucb.data_loader.reshuffle()

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
    print("Average low: {} ({}%)".format(counts[0], 100 * (counts[0] / total)))
    print("Average med: {} ({}%)".format(counts[1], 100 * (counts[1] / total)))
    print("Average high: {} ({}%)".format(counts[2], 100 * (counts[2] / total)))


def main():
    # Note: this sub function (calculating the confidence interval) is from: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h
    def plot_confidence_interval_setup(regret_list, cutoff, num_iters):
        means, lower_bounds, upper_bounds = [], [], []
        for i in range(cutoff - 1, num_iters):
            mean, lower_bound, upper_bound = mean_confidence_interval(regret_list[:i + 1])
            means.append(mean)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        return range(cutoff, num_iters + 1), means, lower_bounds, upper_bounds

    data = pd.read_csv('data/warfarin_clean7.csv')
    features_of_interest = []
    for feat in data.columns:
        for name in FEATURES:
            if name in feat: features_of_interest.append(feat)

    print("Using {} features".format(len(features_of_interest)))
    print(features_of_interest)

    lin_ucb = alg(loader("data/warfarin_clean7.csv", features=features_of_interest, seed=random.randint(1, 100)), use_mlp=True)
    fixed_baseline = FixedDoseBaseline(loader("data/warfarin_clean.csv", features=features_of_interest, seed=random.randint(1, 100)))
    
    cum_regret, avg_regret, avg_accuracy = 0, 0, 0
    regret_list, regret_list_baseline = [], []
    counts = [0,0,0]
    for i in range(NUM_TRIALS):
        reg, avgreg, reg_list = lin_ucb.evaluate_online(return_regret_list=True)
        _, _, reg_list_baseline = fixed_baseline.evaluate_online(return_regret_list=True)
        avg_accuracy += evaluate(lin_ucb.predictions, lin_ucb.data_loader.labels)
        cum_regret += reg
        avg_regret += avgreg
        regret_list += reg_list
        regret_list_baseline += reg_list_baseline
        for pred in lin_ucb.predictions:
            counts[int(pred)] += 1
        lin_ucb.data_loader.reshuffle()

    cum_regret /= NUM_TRIALS
    avg_regret /= NUM_TRIALS
    avg_accuracy /= NUM_TRIALS
    total = sum(counts)

    num_iters = min(len(regret_list), len(regret_list_baseline))
    actions_taken, means, lower_bounds, upper_bounds = plot_confidence_interval_setup(regret_list, 10, num_iters)
    plt.plot(actions_taken, means, lw = 1, color = 'red', alpha = 1, label = "Hybrid Model")
    plt.fill_between(actions_taken, lower_bounds, upper_bounds, color='red', alpha=0.4, label='95% Confidence Interval')
    print("{} {}".format(means[-1], means[-1] - lower_bounds[-1]))


    actions_taken, means, lower_bounds, upper_bounds = plot_confidence_interval_setup(regret_list_baseline, 10, num_iters)
    plt.plot(actions_taken, means, lw=1, color='#539caf', alpha=1, label='Fixed Baseline Average Regret')
    plt.fill_between(actions_taken, lower_bounds, upper_bounds, color='#539caf', alpha=0.4,
                     label='95% Confidence Interval')
    print("{}".format(means[-1] - lower_bounds[-1]))

    plt.xlabel("Number of actions performed")
    plt.ylabel("Average Regret")
    plt.legend()
    plt.show()

    #ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Fit')
    # todo: plot barish graph comparing baseline alg to linear alg
    # todo plot # patients seen vs average regret

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
   # mlp_test()
   #run_s1f()
   #run_modified_ucb()

   #calc_oracle()