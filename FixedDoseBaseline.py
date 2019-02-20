import ModelBaseClass

class FixedDoseBaseline(ModelBaseClass):
    def __init__(self, data_loader):
        super(FixedDoseBaseline, self).__init__(data_loader)

    def next_action(self, patient):
        return self.MED_DOSE

