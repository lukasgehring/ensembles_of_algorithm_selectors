import logging
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import resample

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from math import log, exp

from number_unsolved_instances import NumberUnsolvedInstances


class Boosting:

    def __init__(self, algorithm_name, num_iterations=10, stump=False, singlelearner=False):
        self.algorithm_name = algorithm_name
        self.num_iterations = num_iterations
        self.logger = logging.getLogger("boosting")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.num_models = 0
        self.base_learners = list()
        self.num_base_learner = 10
        self.data_weights = list()
        self.metric = NumberUnsolvedInstances(False)
        self.performances = list()
        self.stump = stump
        self.singlelearner = singlelearner

    def update_weights(self, scenario: ASlibScenario, fold: int, iteration: int, amount_of_training_instances: int):

        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        incorrect_predictions = list()
        total_error = 0

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

            predicted_scores = self.base_learners[iteration].predict(x_test, instance_id)
            if self.metric.evaluate(y_test, predicted_scores, accumulated_feature_time, scenario.algorithm_cutoff_time):
                total_error = total_error + 1
                incorrect_predictions.append(True)
            else:
                incorrect_predictions.append(False)

        total_error = total_error / amount_of_training_instances
        if total_error == 0:
            return False
        performance = 0.5 * log((1 - total_error) / total_error)

        for i, weight in enumerate(self.data_weights):
            if incorrect_predictions[i]:
                self.data_weights[i] = weight * exp(performance)
            else:
                self.data_weights[i] = weight * exp(-performance)
        self.data_weights = self.data_weights / np.sum(self.data_weights)
        self.performances.append(performance)

        return True


    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Start training")
        if amount_of_training_instances == -1:
            amount_of_training_instances = len(scenario.instances)
        self.num_algorithms = len(scenario.algorithms)
        self.data_weights = np.ones(amount_of_training_instances)

        for iteration in range(self.num_iterations):
            print("Start training on iteration", iteration)
            if self.algorithm_name == 'per_algorithm_regressor':
                if self.stump:
                    self.base_learners.append(PerAlgorithmRegressor(data_weights=self.data_weights, stump=True))
                else:
                    self.base_learners.append(PerAlgorithmRegressor(data_weights=self.data_weights))
            elif self.algorithm_name == 'multiclass_algorithm_selector':
                self.base_learners.append(MultiClassAlgorithmSelector(data_weights=self.data_weights))
            elif self.algorithm_name == 'ExponentialSurvivalForest':
                self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential', data_weights=self.data_weights))
            else:
                sys.exit('Wrong base learner for boosting')

            self.base_learners[iteration].fit(scenario, fold, amount_of_training_instances)
            if not self.update_weights(scenario, fold, iteration, amount_of_training_instances):
                break
        print("Finished training")

    def predict(self, features_of_test_instance, instance_id: int):
        if self.singlelearner:
            return self.base_learners[-1].predict(features_of_test_instance, instance_id).flatten()
        else:
            prediction = np.zeros(self.num_algorithms)
            for index, base_learner in enumerate(self.base_learners):
                base_learner_prediction = base_learner.predict(features_of_test_instance, instance_id).flatten()
                base_learner_prediction = [max(base_learner_prediction) / float(i + 0.01) for i in base_learner_prediction]
                base_learner_prediction = [float(i) / sum(base_learner_prediction) for i in base_learner_prediction]
                base_learner_prediction = np.array(base_learner_prediction)
                prediction = prediction + self.performances[index] * base_learner_prediction

            prediction = [max(prediction) / float(i + 0.01) for i in prediction]
            prediction = [float(i) / sum(prediction) for i in prediction]

            return prediction

    def get_name(self):
        name = "boosting_" + self.algorithm_name + "_" + str(self.num_iterations)
        if self.stump:
            name = name + "_stump"
        return name
