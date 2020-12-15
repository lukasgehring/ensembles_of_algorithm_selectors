from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.ensemble import RandomForestClassifier

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import base_learner_performance, split_scenario
from ensembles.stacking import Stacking
from ensembles.validation import split_scenario
import dill
import numpy as np
import pandas as pd
import sys


class StackingOldPreComputed(Stacking):

    def __init__(self, meta_learner_type='per_algorithm_regressor', feature_importances=False):
        Stacking.__init__(self, meta_learner_type=meta_learner_type, cross_validation=False, feature_selection=None, feature_importances=feature_importances, pre_computed=True)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.base_learners.append(self.open_base_learner('per_algorithm_regressor', scenario.scenario, fold))
        self.base_learners.append(self.open_base_learner('sunny', scenario.scenario, fold))
        self.base_learners.append(self.open_base_learner('isac', scenario.scenario, fold))
        self.base_learners.append(self.open_base_learner('satzilla', scenario.scenario, fold))
        self.base_learners.append(self.open_base_learner('expectation', scenario.scenario, fold))
        self.base_learners.append(self.open_base_learner('par10', scenario.scenario, fold))
        self.base_learners.append(self.open_base_learner('multiclass', scenario.scenario, fold))

        Stacking.fit(self, scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        return Stacking.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return Stacking.get_name(self) + '_pre'
