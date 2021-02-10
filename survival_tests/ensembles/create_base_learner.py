import logging

import numpy as np
from scipy.stats import rankdata
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from aslib_scenario.aslib_scenario import ASlibScenario

from ensembles.prediction import predict_with_ranking
from ensembles.validation import base_learner_performance, split_scenario
from par_10_metric import Par10Metric
import sys

import dill

class CreateBaseLearner:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.num_algorithms = 0

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)

        if self.algorithm == 'per_algorithm_regressor':
            base_learner = PerAlgorithmRegressor()
        elif self.algorithm == 'sunny':
            base_learner = SUNNY()
        elif self.algorithm == 'isac':
            base_learner = ISAC()
        elif self.algorithm == 'satzilla':
            base_learner = SATzilla11()
        elif self.algorithm == 'expectation':
            base_learner = SurrogateSurvivalForest(criterion='Expectation')
        elif self.algorithm == 'par10':
            base_learner = SurrogateSurvivalForest(criterion='PAR10')
        elif self.algorithm == 'multiclass':
            base_learner = MultiClassAlgorithmSelector()
        else:
            sys.exit('Wrong base learner')
        base_learner.fit(scenario, fold, amount_of_training_instances)
        self.save_base_learner(base_learner, self.algorithm, scenario.scenario, fold)

    def predict(self, features_of_test_instance, instance_id: int):
        return np.zeros(self.num_algorithms)

    # save base learner for later use
    def save_base_learner(self, base_learner, base_learner_name, scenario_name, fold: int):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'wb') as output:
            dill.dump(base_learner, output)

    def get_name(self):
        name = 'create_base_learner'
        return name
