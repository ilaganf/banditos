'''
eval.py

Function for evaluating the accuracy of warfarin dosing
'''
from __future__ import division

import pandas as pd
import numpy as np

# Note: low/med/high is based on mg/day,
#       but the data is in mg/week
LOW = 0 # 0-3 mg/day
MED = 1 # 3-7 mg/day
HIGH = 2 # >7 mg/day

MED_THRESH = 3
HIGH_THRESH = 7

def evaluate(pred, ground):
    '''
    @pred: predictions, a list of actions where the ith entry
           corresponds to the dosage of the ith patient
    @ground: the ground truth warfarin dosages

    return: classification accuracy
    '''
    bucketed = bucket(ground)
    return np.mean(pred == bucketed)


def bucket(data):
    '''
    @data: array-like, contains ground truth 
           warfarin dosage in mg/week

    return: ndarray with numerical labels for dosage
    '''
    bins = [0, MED_THRESH*7,HIGH_THRESH*7,data.max()]
    return pd.cut(data, bins, include_lowest=True, labels=False)