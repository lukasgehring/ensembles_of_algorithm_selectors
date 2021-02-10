import copy
import logging
import sys
import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario

from ensembles.write_to_database import write_to_database


class GradientBoosting:

    def __init__(self, algorithm_name, max_iterations=200, loss_function='linear'):
        # setup
        self.logger = logging.getLogger("boosting")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.algorithm_name = algorithm_name
        self.max_iterations = max_iterations
        self.loss_function = loss_function

        # attributes
        self.current_iteration = 0
        self.base_learners = list()
        self.start_algorithm = None
        self.beta = list()
        self.data_weights = list()
        self.learning_rate = 0.001
        self.num_algorithm = 0
        self.tmp_predictions = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setup values
        actual_num_training_instances = amount_of_training_instances if amount_of_training_instances != -1 else len(scenario.instances)
        self.num_algorithm = len(scenario.algorithms)

        self.start_algorithm = PerAlgorithmRegressor()
        self.start_algorithm.fit(scenario, fold, amount_of_training_instances)

        residuals = self.get_residuals(scenario, self.start_algorithm, actual_num_training_instances)

        # boosting iterations (stop when avg_loss >= 0.5 or iteration = max_iterations)
        for iteration in range(self.max_iterations):
            for algorithm in range(self.num_algorithm):
                base_learner = DecisionTreeRegressor()
                base_learner.set_params(random_state=fold)

                x_train = scenario.feature_data.to_numpy()

                base_learner.fit(x_train, residuals[algorithm])
                residuals = self.get_residuals(scenario, base_learner, actual_num_training_instances, residuals, algorithm)
                self.base_learners.append(base_learner)

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = self.start_algorithm.predict(features_of_test_instance, instance_id)

        for i, base_learner in enumerate(self.base_learners):
            predictions[i % self.num_algorithm] = predictions[i % self.num_algorithm] + self.learning_rate * base_learner.predict(features_of_test_instance.reshape(1, -1))

        return predictions

    def get_residuals(self, scenario: ASlibScenario, base_learner, amount_of_training_instances: int, old_residuals=None, algorithm=None):
        # get data from original scenario
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None
        num_algorithm = len(scenario.algorithms)

        if old_residuals is None:
            residuals = np.zeros((num_algorithm, amount_of_training_instances))
            self.tmp_predictions = np.zeros((num_algorithm, amount_of_training_instances))
        else:
            residuals = old_residuals

        for instance_id in range(amount_of_training_instances):
            # all instances are for testing
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            #accumulated_feature_time = 0
            #if scenario.feature_cost_data is not None:
            #    feature_time = feature_cost_data[instance_id]
            #    accumulated_feature_time = np.sum(feature_time)

            #contains_non_censored_value = False
            #for y_element in y_test:
            #   if y_element < scenario.algorithm_cutoff_time:
            #       contains_non_censored_value = True
            #if contains_non_censored_value:
            #   num_counted_test_values += 1

            # calculate loss function for each instance
            if old_residuals is None:
                predictions = base_learner.predict(x_test, instance_id).flatten()
                tmp_res = y_test - predictions
                for i, algorith_prediction in enumerate(tmp_res):
                    residuals[i][instance_id] = algorith_prediction
                    self.tmp_predictions[i][instance_id] = predictions[i]
            else:
                prediction = base_learner.predict(x_test.reshape(1, -1))
                residuals[algorithm][instance_id] = y_test[algorithm] - (self.tmp_predictions[algorithm][instance_id] + self.learning_rate * prediction)
                self.tmp_predictions[algorithm][instance_id] = self.tmp_predictions[algorithm][instance_id] + self.learning_rate * residuals[algorithm][instance_id]

        return residuals

    def generate_weighted_sample(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # copy original scenario
        new_scenario = copy.deepcopy(scenario)

        # create weighted sample
        new_scenario.feature_data = scenario.feature_data.sample(amount_of_training_instances, replace=True, weights=self.data_weights, random_state=fold)
        new_scenario.performance_data = scenario.performance_data.sample(amount_of_training_instances, replace=True, weights=self.data_weights, random_state=fold)
        if scenario.feature_cost_data is not None:
            new_scenario.feature_cost_data = scenario.feature_cost_data.sample(amount_of_training_instances, replace=True, weights=self.data_weights, random_state=fold)

        return new_scenario

    def get_name(self):
        name = "adaboostR2_" + self.algorithm_name + "_" + str(self.current_iteration)
        return name
