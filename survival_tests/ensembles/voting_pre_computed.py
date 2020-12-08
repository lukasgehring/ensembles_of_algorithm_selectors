from aslib_scenario.aslib_scenario import ASlibScenario
from ensembles.voting import Voting
import dill
import sys


class VotingPreComputed(Voting):

    def __init__(self, ranking=False, weighting=False, cross_validation=False, base_learner=None,
                 rank_method='average'):
        Voting.__init__(self, base_learner=base_learner, ranking=ranking)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)
        if 1 in self.base_learner:
            self.trained_models.append(self.open_base_learner('per_algorithm_regressor', scenario.scenario, fold))
        if 2 in self.base_learner:
            self.trained_models.append(self.open_base_learner('sunny', scenario.scenario, fold))
        if 3 in self.base_learner:
            self.trained_models.append(self.open_base_learner('isac', scenario.scenario, fold))
        if 4 in self.base_learner:
            self.trained_models.append(self.open_base_learner('satzilla', scenario.scenario, fold))
        if 5 in self.base_learner:
            self.trained_models.append(self.open_base_learner('expectation', scenario.scenario, fold))
        if 6 in self.base_learner:
            self.trained_models.append(self.open_base_learner('par10', scenario.scenario, fold))
        if 7 in self.base_learner:
            self.trained_models.append(self.open_base_learner('multiclass', scenario.scenario, fold))

    def predict(self, features_of_test_instance, instance_id: int):
        return Voting.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return Voting.get_name(self) + '_pre'
