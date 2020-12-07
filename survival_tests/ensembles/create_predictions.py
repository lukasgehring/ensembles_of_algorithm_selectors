import logging

import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
import os

import dill

class CreatePrediction:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.num_algorithms = 0
        self.predictions = dict()
        self.scenario = None
        self.base_learner = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.scenario = scenario
        self.num_algorithms = len(scenario.algorithms)
        self.base_learner = self.open_base_learner(self.algorithm, scenario.scenario, fold)

    def predict(self, features_of_test_instance, instance_id: int):
        self.predictions[str(features_of_test_instance)] = self.base_learner.predict(features_of_test_instance, instance_id)
        self.save_prediction()
        return np.zeros(self.num_algorithms)

    # save base learner for later use
    def save_prediction(self):
        file_name = 'pre_computed_prediction/' + self.algorithm + self.scenario.scenario
        self.delete_file(file_name)
        with open(file_name, 'wb') as output:
            dill.dump(self.predictions, output)

    def delete_file(self, name):
        if os.path.exists(name):
            os.remove(name)
        else:
            print('file does not exist')

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        name = 'create_base_learner'
        return name
