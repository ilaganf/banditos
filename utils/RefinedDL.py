'''
Refined dataloader
'''
import pandas as pd

from utils.DataLoader import DataLoader

GROUND = "Therapeutic Dose of Warfarin"

class RefinedDL(DataLoader):
    '''
    Data loader that uses features based on a feature list
    '''

    def __init__(self, filename_csv, features=None, seed=420):
        '''
        Mostly same as regular DataLoader, with main difference being features

        @param features: can be int or iterable. If int, only uses that many columns
                         of the data. If iterable, uses the features specified in the iterable
        '''
        raw = pd.read_csv(filename_csv)
        self.data = raw.sample(frac=1, replace=False, random_state=seed)
        self.labels = self.data[GROUND]
        self.data.drop(GROUND, axis=1, inplace=True)
        if features:
            self.data = self.data[:,:features] if type(features) is int else self.data.loc[:,features]
        self.ind = 0
