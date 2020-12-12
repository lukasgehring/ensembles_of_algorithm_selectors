from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.ensemble import RandomForestClassifier

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import base_learner_performance, split_scenario
from ensembles.stacking_new import StackingNew
from ensembles.validation import split_scenario
import dill
import numpy as np
import pandas as pd
import sys


class StackingPreComputed(StackingNew):

    def __init__(self, meta_learner_type='random_forest_classifier', cross_validation=False, feature_selection=None, base_learner=None, feature_type='standard'):
        StackingNew.__init__(self, meta_learner_type=meta_learner_type, cross_validation=False, feature_selection=None, base_learner=base_learner, pre_computed=True, type=feature_type)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        if 1 in self.base_learner:
            self.base_learners.append(self.open_base_learner('per_algorithm_regressor', scenario.scenario, fold))
        if 2 in self.base_learner:
            self.base_learners.append(self.open_base_learner('sunny', scenario.scenario, fold))
        if 3 in self.base_learner:
            self.base_learners.append(self.open_base_learner('isac', scenario.scenario, fold))
        if 4 in self.base_learner:
            self.base_learners.append(self.open_base_learner('satzilla', scenario.scenario, fold))
        if 5 in self.base_learner:
            self.base_learners.append(self.open_base_learner('expectation', scenario.scenario, fold))
        if 6 in self.base_learner:
            self.base_learners.append(self.open_base_learner('par10', scenario.scenario, fold))
        if 7 in self.base_learner:
            self.base_learners.append(self.open_base_learner('multiclass', scenario.scenario, fold))

        StackingNew.fit(self, scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        return StackingNew.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return StackingNew.get_name(self) + '_pre'
