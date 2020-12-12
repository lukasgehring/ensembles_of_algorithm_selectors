import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import split_scenario
import copy

from ensembles.validation import base_learner_performance


class StackingNew:

    def __init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, base_learner=None):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.cross_validation = cross_validation
        self.feature_selection = feature_selection
        self.meta_learner_type = meta_learner_type
        self.base_learner = base_learner


        # attributes
        self.meta_learner = None
        self.base_learners = list()
        self.num_algorithms = 0

    def create_base_learner(self):
        self.base_learners = list()

        if 1 in self.base_learner:
            self.base_learners.append(PerAlgorithmRegressor())
        if 2 in self.base_learner:
            self.base_learners.append(SUNNY())
        if 3 in self.base_learner:
            self.base_learners.append(ISAC())
        if 4 in self.base_learner:
            self.base_learners.append(SATzilla11())
        if 5 in self.base_learner:
            self.base_learners.append(SurrogateSurvivalForest(criterion='Expectation'))
        if 6 in self.base_learner:
            self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        if 7 in self.base_learner:
            self.base_learners.append(MultiClassAlgorithmSelector())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setup
        self.create_base_learner()
        self.num_algorithms = len(scenario.algorithms)
        feature_data = scenario.feature_data.to_numpy()
        num_instances = len(feature_data)
        x_train = np.zeros((num_instances, self.num_algorithms))

        for i, base_learner in enumerate(self.base_learners):
            base_learner.fit(scenario, fold, amount_of_training_instances)
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
        # get all predictions from the base learners
        new_feature_data = np.zeros(self.num_algorithms)

        for base_learner in self.base_learners:
            # create new feature data
            algorithm_prediction = np.argmin(base_learner.predict(features_of_test_instance, instance_id))
            new_feature_data[algorithm_prediction] = new_feature_data[algorithm_prediction] + 1

        features_of_test_instance = new_feature_data.reshape(1, -1)

        prediction = self.meta_learner.predict(features_of_test_instance)

        final_prediction = np.ones(self.num_algorithms)
        final_prediction[prediction] = 0

        return final_prediction

    def get_name(self):
        name = "stacking_new_" + self.meta_learner_type
        if self.cross_validation:
            name = name + "_cross_validation"
        if self.feature_selection is not None:
            name = name + "_" + self.feature_selection

        return name
