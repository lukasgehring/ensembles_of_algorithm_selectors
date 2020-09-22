import logging

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from math import log, exp

from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from number_unsolved_instances import NumberUnsolvedInstances


class Stacking:

    def __init__(self):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.num_models = 0
        self.base_learners = list()
        self.num_base_learner = 0
        self.metric = NumberUnsolvedInstances(False)

    def create_base_learner(self):
        self.base_learners.append(PerAlgorithmRegressor())
        self.base_learners.append(SUNNY())
        self.base_learners.append(ISAC())
        self.base_learners.append(SATzilla11())
        self.base_learners.append(MultiClassAlgorithmSelector())
        self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential'))
        self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        self.num_base_learner = len(self.base_learners)

    def base_learner_predict(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        feature_data = scenario.feature_data.to_numpy()

        predictions = list()

        for i, base_learner in enumerate(self.base_learners):
            predictions.append(list())
            for instance_id in range(amount_of_training_instances):
                x_test = feature_data[instance_id]
                predictions[i].append(base_learner.predict(x_test, instance_id))

        return predictions

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()

        for base_learner in self.trained_models:
            base_learner.fit(scenario, fold, amount_of_training_instances)

        if amount_of_training_instances == -1:
            amount_of_training_instances = len(scenario.instances)
        self.num_algorithms = len(scenario.algorithms)

        predictions = self.base_learner_predict(scenario, fold, amount_of_training_instances)
        #[1,2,3][4,3,1][3,4,2][1,2,4]

        meta_learner = RandomForestRegressor(max_depth=2, random_state=fold)
        meta_learner.fit()

    def predict(self, features_of_test_instance, instance_id: int):
        return self.base_learners[-1].predict(features_of_test_instance, instance_id)

    # TODO: What does this method?
    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(performance_data, axis=0)) if num_instances > 0 else np.size(
            performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def get_name(self):
        return "stacking"
