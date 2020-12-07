import logging
import sys
import math

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import resample

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from math import log, exp

from number_unsolved_instances import NumberUnsolvedInstances


class AdaboostR2:

    def __init__(self, algorithm_name, num_iterations=10):
        self.algorithm_name = algorithm_name
        self.num_iterations = num_iterations
        self.logger = logging.getLogger("boosting")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.num_models = 0
        self.base_learners = list()
        self.beta = list()
        self.data_weights = list()
        self.metric = NumberUnsolvedInstances(False)
        self.performances = list()

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        actual_num_training_instances = amount_of_training_instances if amount_of_training_instances != -1 else len(scenario.instances)
        self.num_algorithms = len(scenario.algorithms)
        self.data_weights = np.ones(actual_num_training_instances) / actual_num_training_instances

        for iteration in range(self.num_iterations):
            if self.algorithm_name == 'per_algorithm_regressor':
                self.base_learners.append(PerAlgorithmRegressor(data_weights=self.data_weights))
            elif self.algorithm_name == 'multiclass_algorithm_selector':
                self.base_learners.append(MultiClassAlgorithmSelector(data_weights=self.data_weights))
            elif self.algorithm_name == 'ExponentialSurvivalForest':
                self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential', data_weights=self.data_weights))
            else:
                sys.exit('Wrong base learner for boosting')

            self.base_learners[iteration].fit(scenario, fold, amount_of_training_instances)
            self.update_weights(scenario, self.base_learners[iteration], actual_num_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        y_predictions = list()
        for base_learner, beta in zip(self.base_learners, self.beta):
            y_predictions.append((np.amin(base_learner.predict(features_of_test_instance, instance_id)), beta, base_learner))
        y_predictions.sort(key=lambda x: x[0])

        lower_bound = 0.0
        for beta in y_predictions:
            lower_bound = lower_bound + math.log(1/beta[1])
        lower_bound = lower_bound * 0.5

        beta_sum = 0.0
        for t, beta in enumerate(y_predictions):
            beta_sum = beta_sum + math.log(1 / beta[1])
            if beta_sum >= lower_bound:
                prediction = y_predictions[t][2].predict(features_of_test_instance, instance_id)
                return prediction

    def update_weights(self, scenario: ASlibScenario, base_learner, amount_of_training_instances: int):
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        temp_loss = list()

        for instance_id in range(amount_of_training_instances):
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if scenario.feature_cost_data is not None:
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            # contains_non_censored_value = False
            # for y_element in y_test:
            #    if y_element < test_scenario.algorithm_cutoff_time:
            #        contains_non_censored_value = True
            # if contains_non_censored_value:
            #    num_counted_test_values += 1

            predictions = base_learner.predict(x_test, instance_id)
            index = np.argmin(y_test)
            temp_loss.append(abs(predictions.flatten()[index] - y_test[index]))
        loss = temp_loss / np.amax(temp_loss)
        avg_loss = sum(loss * self.data_weights)
        beta = avg_loss / (1 - avg_loss)
        self.beta.append(beta)
        self.data_weights = self.data_weights * beta**(1-loss)

    def get_name(self):
        name = "adaboostR2_" + self.algorithm_name + "_" + str(self.num_iterations)
        return name
