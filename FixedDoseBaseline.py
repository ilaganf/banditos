from ModelBaseClass import ModelBaseClass
from utils.DataLoader import DataLoader

class FixedDoseBaseline(ModelBaseClass):
    def __init__(self, data_loader):
        super(FixedDoseBaseline, self).__init__(data_loader)

    def next_action(self, patient):
        return self.MED_DOSE


if __name__ == '__main__':
    baseline = FixedDoseBaseline(DataLoader("data/warfarin_clean.csv"))
    cum_regret, avg_regret = baseline.evaluate_online()
    print("cum_regret {}, avg_regret {}".format(cum_regret, avg_regret))

