
import pandas as pd
import numpy as np

GROUND = "Therapeutic Dose of Warfarin"

class DataLoader():
    def __init__(self, filename_csv):
        """
        Converts the csv into a cleaned pandas dataframe where
        categorical variables are turned into indicators
        """
        raw = pd.read_csv(filename_csv)
        self.labels = raw[GROUND]
        raw.drop(GROUND, axis=1, inplace=True)
        self.data = pd.get_dummies(raw, prefix='indic')

    
    def sample_next_patient(self):
        """
        Samples without replacement from the patients in the dataset 

        @returns patient, row of features for a given patient, or None if all patients already sampled 
        """
        pass