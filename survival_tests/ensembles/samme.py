import logging
import sys
import math

import numpy as np

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario

from ensembles.write_to_database import write_to_database
from number_unsolved_instances import NumberUnsolvedInstances


class SAMME:

    def __init__(self, algorithm_name, num_iterations=20):
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
        self.current_iteration = 0

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        actual_num_training_instances = amount_of_training_instances if amount_of_training_instances != -1 else len(scenario.instances)
        self.num_algorithms = len(scenario.algorithms)
        self.data_weights = np.ones(actual_num_training_instances) / actual_num_training_instances

        print(sum(self.data_weights))

        for iteration in range(self.num_iterations):
            if self.algorithm_name == 'per_algorithm_regressor':
                self.base_learners.append(PerAlgorithmRegressor(data_weights=self.data_weights))
                #self.base_learners.append(PerAlgorithmRegressor())
            elif self.algorithm_name == 'multiclass_algorithm_selector':
                self.base_learners.append(MultiClassAlgorithmSelector(data_weights=self.data_weights))
            elif self.algorithm_name == 'ExponentialSurvivalForest':
                self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential', data_weights=self.data_weights))
            else:
                sys.exit('Wrong base learner for boosting')

            self.base_learners[iteration].fit(scenario, fold, amount_of_training_instances)
            if not self.update_weights(scenario, self.base_learners[iteration], actual_num_training_instances):
                break

            self.current_iteration = self.current_iteration + 1
            write_to_database(scenario, self, fold)

    def predict(self, features_of_test_instance, instance_id: int):
        confidence = np.zeros(self.num_algorithms)

        #for base_learner, beta in zip(self.base_learners, self.beta):
        #    predicted_algorithm = np.argmin(base_learner.predict(features_of_test_instance, instance_id))
        #    confidence[predicted_algorithm] = confidence[predicted_algorithm] + beta

        confidence = self.base_learners[len(self.base_learners) - 1].predict(features_of_test_instance, instance_id)

        final_prediction = np.ones(self.num_algorithms)
        #TODO: change to max
        final_prediction[np.argmin(confidence)] = 0
        return final_prediction

    def update_weights(self, scenario: ASlibScenario, base_learner, amount_of_training_instances: int):
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        err = 0
        is_correct = list()

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
            y_algorithm = np.argmin(y_test)
            predicted_algorithm = np.argmin(predictions)
            if y_algorithm != predicted_algorithm:
                err = err + self.data_weights[instance_id]
                is_correct.append(False)
            else:
                is_correct.append(True)

        err = err / sum(self.data_weights)

        if(1 - err <= 1 / self.num_algorithms):
            return False

        beta = math.log((1 - err) / err) + math.log(self.num_algorithms - 1)
        self.beta.append(beta)

        for i in range(amount_of_training_instances):
            if not is_correct[i]:
                self.data_weights[i] = self.data_weights[i] * math.exp(beta)

        self.data_weights = self.data_weights / sum(self.data_weights)
        return True

    def get_name(self):
        name = "SAMME_" + self.algorithm_name + "_" + str(self.current_iteration)
        return name
