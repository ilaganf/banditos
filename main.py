'''
main.py

Main entrypoint for bandit algorithm on Warfarin dataset
'''
import random

import pandas as pd

from LinUCB import LinUCB as alg
# from LassoUCB import LassoUCB as alg
# from FixedDoseBaseline import FixedDoseBaseline as alg
from utils.RefinedDL import RefinedDL as loader

from utils.eval import evaluate
'''
fixed dose: -2319 regret, 53.36% accuracy

age, weight, height, gender, alpha=2.38: -2330 regret, 53.28% accuracy

'''
FEATURES = ['Weight','indic_male', 'indic_female', 'Age', 'Height']
# FEATURES = ['Weight', 'indic_male', 'indic_female', 'Age', 'Height', \
            # 'indic_*', 'indic_1', 'indic_2', 'indic_3', 'indic_A', 'indic_C',\
            # 'indic_G', 'indic_T', 'Smoker', 'Acetaminophen', 'Asian', 'Black', 'White', 'Race']
FEATURES = ['indic_male','indic_female']

NUM_TRIALS = 1

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

    print("Results (averaged over {} trials)".format(NUM_TRIALS))
    # print("Alpha = ", lin_ucb.alpha)
    print("Cumulative Regret {}, Average Regret {}".format(cum_regret, avg_regret))
    print("Accuracy: ", avg_accuracy)
    print("Average low: {} ({}%)".format(counts[0], 100*(counts[0]/total)))
    print("Average med: {} ({}%)".format(counts[1], 100*(counts[1]/total)))
    print("Average high: {} ({}%)".format(counts[2], 100*(counts[2]/total)))

if __name__ == '__main__':
    main()