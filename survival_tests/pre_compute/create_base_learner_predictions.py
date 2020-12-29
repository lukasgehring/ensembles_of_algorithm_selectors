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

from ensembles.validation import split_scenario
from pre_compute.pickle_loader import save_pickle


class CreateBaseLearnerPrediction:

    def __init__(self, algorithm, for_cross_validation=False):
        self.algorithm = algorithm
        self.num_algorithms = 0
        self.base_learner = None
        self.scenario_name = ''
        self.fold = 0
        self.pred = {}
        self.for_cross_validation = for_cross_validation

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

        if self.for_cross_validation:

            num_instances = len(scenario.instances)

            feature_data = scenario.feature_data.to_numpy()

            instance_counter = 0

            predictions = np.zeros((num_instances, self.num_algorithms))

            for sub_fold in range(1, 11):
                test_scenario, training_scenario = split_scenario(scenario, sub_fold, num_instances)

                # train base learner
                self.base_learner.fit(training_scenario, fold, amount_of_training_instances)

                # create new feature data
                for instance_number in range(instance_counter, instance_counter + len(test_scenario.instances)):
                    prediction = self.base_learner.predict(feature_data[instance_number], instance_number).flatten()
                    predictions[instance_number] = prediction

                instance_counter = instance_counter + len(test_scenario.instances)

            save_pickle(filename='predictions/cross_validation_' + self.base_learner.get_name() + '_' + self.scenario_name + '_' + str(self.fold), data=predictions)
        else:
            self.base_learner.fit(scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):

        if not self.for_cross_validation:
            self.pred[str(features_of_test_instance)] = self.base_learner.predict(features_of_test_instance,
                                                                                  instance_id).flatten()
            save_pickle(filename='predictions/' + self.base_learner.get_name() + '_' + self.scenario_name + '_' + str(self.fold), data=self.pred)

        return np.zeros(self.num_algorithms)

    def get_name(self):
        name = 'create_base_learner'
        return name
