from aslib_scenario.aslib_scenario import ASlibScenario

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import base_learner_performance, split_scenario
from ensembles.voting import Voting
import dill
import numpy as np
import sys


class VotingPreComputed(Voting):

    def __init__(self, ranking=False, weighting=False, cross_validation=False, base_learner=None,
                 rank_method='average'):
        Voting.__init__(self, base_learner=base_learner, ranking=ranking, weighting=weighting, cross_validation=cross_validation, rank_method=rank_method, pre_computed=True)

    def cross_validation_model_backup(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.trained_models_backup = list()

        if 1 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('per_algorithm_regressor', scenario.scenario, fold))
        if 2 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('sunny', scenario.scenario, fold))
        if 3 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('isac', scenario.scenario, fold))
        if 4 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('satzilla', scenario.scenario, fold))
        if 5 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('expectation', scenario.scenario, fold))
        if 6 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('par10', scenario.scenario, fold))
        if 7 in self.base_learner:
            self.trained_models_backup.append(self.open_base_learner('multiclass', scenario.scenario, fold))

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.trained_models = list()

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

        if self.cross_validation:
            self.cross_validation_model_backup(scenario, fold, amount_of_training_instances)

        Voting.fit(self, scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        return Voting.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return Voting.get_name(self) + '_pre'
