from ModelBaseClass import ModelBaseClass
import math
import numpy as np

class S1fBaseline(ModelBaseClass):
    """
    @param data_loader See DataLoader.py
    @param relevant_col_indices Columns corresponding to (in order): age, weight, height, asian, black or african american,
    missing or mixed race, Amiodarone status, Enzyme inducer status
    """
    def __init__(self, data_loader):
        #"Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American",
          #           "Unknown or mixed race", "Amiodarone (Cordarone)", "Enzyme inducer status"
        super(S1fBaseline, self).__init__(data_loader)
        # self.relevant_col_indices = relevant_col_indices
        self.weights = np.array([-0.2546, 0.0118, 0.0134, -0.6752, 0.4060, 0.0443, -0.5695, 1.2799]) # removed  for enzyme inducer status since not in dataset

    def next_action(self, patient):
        x_t = patient.values.T
        x_t = np.squeeze(x_t, 1)

        # aggregate remaining attributes. if any are true, has enzyme inducer status
        vals = x_t[:8]
        has_drug = sum(val for val in vals[7:]) > 0
        if has_drug:
            vals[7] = 1
        else:
            vals[7] = 0

        sqrt_weekly_dose = 4.0376
        sqrt_weekly_dose += np.dot(vals, self.weights)
        daily_dose = (sqrt_weekly_dose ** 2) / 7
        return self.dose_to_action(daily_dose)

