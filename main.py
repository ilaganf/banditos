'''
main.py

Main entrypoint for bandit algorithm on Warfarin dataset
'''
import pandas as pd

from LinUCB import LinUCB as alg
from utils.RefinedDL import RefinedDL as loader

'''
fixed dose: -2319

ethnicity and weight: -2403
ethnicity, weight, gender: -2342

'''
# FEATURES = ['indic', 'Height', 'Weight', 'Diabetes', 'Ethnicity']
FEATURES = ['Weight','Ethnicity', 'indic_male', 'indic_female', 'Age', 'Tylenol']

def main():
    data = pd.read_csv('data/warfarin_clean.csv')
    features_of_interest = []
    for feat in data.columns:
        for name in FEATURES:
            if name in feat: features_of_interest.append(feat)

    print("Using {} features".format(len(features_of_interest)))
    print(features_of_interest)

    lin_ucb = alg(loader("data/warfarin_clean.csv", features_of_interest))
    cum_regret, avg_regret = lin_ucb.evaluate_online()
    print("cum_regret {}, avg_regret {}".format(cum_regret, avg_regret))

if __name__ == '__main__':
    main()