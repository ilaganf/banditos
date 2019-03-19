import random

class ModelBaseClass():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.LOW_DOSE, self.MED_DOSE, self.HIGH_DOSE = 0, 1, 2
        self.actions = [self.LOW_DOSE, self.MED_DOSE, self.HIGH_DOSE]

    def next_action(self, patient):
        pass

    def reward_for_timestep(self):
        pass

    def update_model(self, patient, actual_action, ideal_action):
        pass

    """
    @param dose, daily dose of medicine in mg 
    @returns action 
    
    """
    def dose_to_action(self, dose):
        if dose < 3:
            return self.LOW_DOSE
        elif dose <= 7:
            return self.MED_DOSE
        else:
            return self.HIGH_DOSE

    def ideal_action(self, ideal_mg_per_week):
        return self.dose_to_action(ideal_mg_per_week / 7.0)

    def select_action(self, p):
        max_r = max(p)[0]
        cands = [a for r, a in p if r == max_r]
        return random.choice(cands)

    def evaluate_online(self):
        """
        Simulates and evaluates online learning model with samples from data_loader
        
        @returns cumulative_regret, avg_regret 
        """
        cumulative_regret = 0.0
        patient, ideal_mg_per_week = self.data_loader.sample_next_patient()
        self.predictions = []
        self.times_action_taken = [1.0] * len(self.actions)
        self.seen_data = [[] for _ in range(len(self.actions))]
        ideal_action_counts = [0.0] * 3
        while patient is not None:
            # reward is 0 if correct action, -1 otherwise
            ideal_action = self.ideal_action(ideal_mg_per_week)
            actual_action = self.next_action(patient)
            ideal_action_counts[ideal_action] += 1
            self.times_action_taken[actual_action] += 1
            self.update_model(patient, actual_action, ideal_action)
            if ideal_action != actual_action:
                cumulative_regret += -1
            self.predictions.append(actual_action)
            patient, ideal_mg_per_week = self.data_loader.sample_next_patient()
        print("ideal action counts {}".format(ideal_action_counts))
        return cumulative_regret, cumulative_regret / self.data_loader.num_samples()