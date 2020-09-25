import logging

import numpy as np
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from number_unsolved_instances import NumberUnsolvedInstances


class Stacking:

    def __init__(self):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        self.meta_learner = PerAlgorithmRegressor()
        self.base_learners = list()

        self.num_algorithms = 0
        self.num_models = 0

    def create_base_learner(self):
        self.base_learners.append(PerAlgorithmRegressor())
        self.base_learners.append(SUNNY())
        self.base_learners.append(ISAC())
        self.base_learners.append(SATzilla11())
        self.base_learners.append(MultiClassAlgorithmSelector())
        self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential'))
        self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()
        feature_data = scenario.feature_data.to_numpy()

        # train base learner
        for base_learner in self.base_learners:
            base_learner.fit(scenario, fold, amount_of_training_instances)
            for instance_id in range(amount_of_training_instances):
                np.concatenate((feature_data[instance_id], base_learner.predict(feature_data[instance_id], instance_id).flatten()))

        # add predictions to the features of the instances
        scenario.feature_data = feature_data

        # meta learner training
        self.meta_learner = PerAlgorithmRegressor()
        self.meta_learner.fit(scenario, fold, amount_of_training_instances)


    def predict(self, features_of_test_instance, instance_id: int):
        # get all predictions from the base learners
        base_learner_predictions = list()
        for base_learner in self.base_learners:
            base_learner_predictions.append(base_learner.predict(features_of_test_instance, instance_id))

        # add the predictions to the features of the instance
        for i in range(len(self.base_learners)):
            np.concatenate((features_of_test_instance, base_learner_predictions[i].flatten()))

        # final prediction
        return self.meta_learner.predict(features_of_test_instance, instance_id)

    def get_name(self):
        return "stacking"
