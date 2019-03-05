
import pandas as pd
import numpy as np

GROUND = "Therapeutic Dose of Warfarin"

class DataLoader():
    def __init__(self, filename_csv, seed=42):
        """
        Converts the csv into a cleaned pandas dataframe where
        categorical variables are turned into indicators
        """
        raw = pd.read_csv(filename_csv)
        # shuffles dataset
        self.data = pd.get_dummies(raw, prefix='indic').sample(frac=1, replace=False, random_state=seed)
        self.labels = self.data[GROUND]
        self.data.drop(GROUND, axis=1, inplace=True)
        self.ind = 0

    # @returns the total number of data points in the dataset
    def num_samples(self):
        return len(self.data)

    def sample_next_patient(self):
        """
        Samples without replacement from the patients in the dataset 

        @returns patient, row of features for a given patient, or None if all patients already sampled 
        """
        # TODO: randomly sample instead of iterate through each row in file
        self.ind += 1
        return (None, None) if self.ind > len(self.data) else (self.data[self.ind - 1:self.ind], self.labels[self.ind - 1])