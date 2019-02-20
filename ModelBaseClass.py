
class ModelBaseClass():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.LOW_DOSE, self.MED_DOSE, self.HIGH_DOSE = 0, 1, 2

    def next_action(self, patient):
        pass

    def reward_for_timestep(self):
        pass

    """
    Simulates and evaluates online learning model with samples from data_loader
    
    @returns cumulative_regret, avg_regret 
    """
    def evaluate_online(self):
        pass