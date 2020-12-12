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

    def __init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, base_learner=None):
        StackingNew.__init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, base_learner=base_learner)

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

        self.num_algorithms = len(scenario.algorithms)
        feature_data = scenario.feature_data.to_numpy()
        num_instances = len(feature_data)
        x_train = np.zeros((num_instances, self.num_algorithms))

        for i, base_learner in enumerate(self.base_learners):
            for instance_number, x_test in enumerate(feature_data):
                algorithm_prediction = np.argmin(base_learner.predict(x_test, instance_number))
                x_train[instance_number][algorithm_prediction] = x_train[instance_number][algorithm_prediction] + 1

        self.meta_learner = RandomForestClassifier(n_jobs=1, n_estimators=100)
        self.meta_learner.set_params(random_state=fold)
        y_train = list()
        for data in scenario.performance_data.to_numpy():
            y_train.append(np.argmin(data))
        self.meta_learner.fit(x_train, y_train)

    def predict(self, features_of_test_instance, instance_id: int):
        return StackingNew.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return StackingNew.get_name(self) + '_pre'
