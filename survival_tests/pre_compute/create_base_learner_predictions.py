import numpy as np
import sys
import pickle

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from aslib_scenario.aslib_scenario import ASlibScenario

from pre_compute.pickle_loader import save_prediction


class CreateBaseLearnerPrediction:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.num_algorithms = 0
        self.base_learner = None
        self.scenario_name = ''
        self.fold = 0
        self.pred = {}

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)
        self.scenario_name = scenario.scenario
        self.fold = fold

        if self.algorithm == 'per_algorithm_regressor':
            self.base_learner = PerAlgorithmRegressor()
        elif self.algorithm == 'sunny':
            self.base_learner = SUNNY()
        elif self.algorithm == 'isac':
            self.base_learner = ISAC()
        elif self.algorithm == 'satzilla':
            self.base_learner = SATzilla11()
        elif self.algorithm == 'expectation':
            self.base_learner = SurrogateSurvivalForest(criterion='Expectation')
        elif self.algorithm == 'par10':
            self.base_learner = SurrogateSurvivalForest(criterion='PAR10')
        elif self.algorithm == 'multiclass':
            self.base_learner = MultiClassAlgorithmSelector()
        else:
            sys.exit('Wrong base learner')
        self.base_learner.fit(scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        save_prediction(self, self.algorithm, features_of_test_instance, self.base_learner.predict(features_of_test_instance, instance_id))
        return np.zeros(self.num_algorithms)

    def get_name(self):
        name = 'create_base_learner'
        return name
