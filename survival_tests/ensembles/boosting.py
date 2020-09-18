import logging

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import resample
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario

from number_unsolved_instances import NumberUnsolvedInstances


class Boosting:

    def __init__(self):
        self.logger = logging.getLogger("boosting")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.num_models = 0
        self.base_learners = list()
        self.num_base_learner = 10
        self.data_weights = list()
        self.metric = NumberUnsolvedInstances(False)

    def update_weights(self, scenario: ASlibScenario, fold: int, iteration: int, amount_of_training_instances: int):
        test_scenario, train_scenario = scenario.get_split(indx=fold)

        feature_data = train_scenario.feature_data.to_numpy()
        performance_data = train_scenario.performance_data.to_numpy()
        feature_cost_data = train_scenario.feature_cost_data.to_numpy() if train_scenario.feature_cost_data is not None else None

        for instance_id in range(amount_of_training_instances):
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if train_scenario.feature_cost_data is not None:
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
                self.data_weights[instance_id] = self.data_weights[instance_id] * 2
            else:
                self.data_weights[instance_id] = self.data_weights[instance_id] / 2

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))

        test_scenario, train_scenario = scenario.get_split(indx=fold)

        if amount_of_training_instances == -1:
            amount_of_training_instances = len(train_scenario.instances)
        self.num_algorithms = len(scenario.algorithms)

        self.data_weights = np.ones(amount_of_training_instances)

        for iteration in range(10):
            print(self.data_weights)
            self.base_learners.append(PerAlgorithmRegressor(data_weights=self.data_weights))
            self.base_learners[iteration].fit(scenario, fold, amount_of_training_instances)
            self.update_weights(scenario, fold, iteration, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        return self.base_learners[-1].predict(features_of_test_instance, instance_id)

    # TODO: What does this method?
    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(performance_data, axis=0)) if num_instances > 0 else np.size(
            performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def get_name(self):
        return "boosting"
