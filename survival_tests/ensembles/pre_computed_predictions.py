from aslib_scenario.aslib_scenario import ASlibScenario
from ensembles.voting import Voting
import dill
import sys
import numpy as np


class PreComputedPredictions:

    def __init__(self, algorithm=None):
        self.predictions = None
        self.algorithm = algorithm
        self.scenario = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.scenario = scenario
        if self.predictions is None:
            self.predictions.append(self.open_predictions('per_algorithm_regressor'))
            self.predictions.append(self.open_predictions('sunny'))
            self.predictions.append(self.open_predictions('isac'))
            self.predictions.append(self.open_predictions('satzilla'))
            self.predictions.append(self.open_predictions('expectation'))
            self.predictions.append(self.open_predictions('par10'))

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros(self.num_algorithms)
        for i, prediction in enumerate(self.predictions):
            # get prediction of base learner and find prediction (lowest value)
            base_prediction = prediction[str(features_of_test_instance)]
            index_of_minimum = np.where(base_prediction == min(base_prediction))


            predictions[index_of_minimum] = predictions[index_of_minimum] + 1

        return 1 - predictions / sum(predictions)
        return self.predictions[str(features_of_test_instance)]

    # open pre computed base learner
    def open_predictions(self, algorithm):
        file_name = 'pre_computed_prediction/' + algorithm + self.scenario.scenario
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return 'pre_computed_predictoons'
