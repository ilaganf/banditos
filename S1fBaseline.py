import ModelBaseClass
import math

class S1fBaseline(ModelBaseClass):
    """
    @param data_loader See DataLoader.py
    @param relevant_col_indices Columns corresponding to (in order): age, weight, height, asian, black or african american,
    missing or mixed race, Enzyme inducer status, Amiodarone status
    """
    # todo: verify units of data in csv are what the weights in pdf wants
    def __init__(self, data_loader, relevant_col_indices=list(range(8))):
        super(S1fBaseline, self).__init__(data_loader)
        self.relevant_col_indices = relevant_col_indices
        self.weights = [-0.2546, 0.0118, 0.0134, -0.6752, 0.4060, 0.0443, 1.2799, -0.5695]

    def next_action(self, patient):
        sqrt_weekly_dose = 4.0376
        for i in self.relevant_col_indices:
            sqrt_weekly_dose += self.weights[i] * patient[i]
        daily_dose = (sqrt_weekly_dose ** 2) * 7
        return daily_dose

