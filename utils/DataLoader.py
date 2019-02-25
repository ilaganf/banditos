
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

    
    def sample_next_patient(self):
        """
        Samples without replacement from the patients in the dataset 

        @returns patient, row of features for a given patient, or None if all patients already sampled 
        """
        self.ind += 1
        return None if self.ind > len(self.data) else self.data[self.ind-1]