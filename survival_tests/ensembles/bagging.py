import copy
import logging

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import resample

from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario


class Bagging:

    def __init__(self, num_base_learner: int, base_learner=PerAlgorithmRegressor()):
        self.logger = logging.getLogger("bagging")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.base_learner = base_learner
        self.base_learners = list()
        self.num_base_learner = num_base_learner

    # generate number_of_samples bootstrap samples from the scenario and returns them in a list
    def generate_bootstrap_sample(self, scenario: ASlibScenario, fold: int, number_of_samples: int):
        bootstrap_samples = list()
        sample_size = len(scenario.instances)
        random_seed = (fold - 1) * number_of_samples

        for i in range(number_of_samples):
            feature_data_sample = scenario.feature_data.sample(sample_size, replace=True, random_state=random_seed)
            performance_data_sample = scenario.performance_data.sample(sample_size, replace=True, random_state=random_seed)
            runstatus_data_sample = scenario.runstatus_data.sample(sample_size, replace=True, random_state=random_seed)
            feature_runstatus_data_sample = scenario.feature_runstatus_data.sample(sample_size, replace=True, random_state=random_seed)
            if scenario.feature_cost_data is not None:
                feature_cost_data_sample = scenario.feature_cost_data.sample(sample_size, replace=True, random_state=random_seed)
            else:
                feature_cost_data_sample = None

            bootstrap_samples.append((feature_data_sample, performance_data_sample, runstatus_data_sample, feature_runstatus_data_sample, feature_cost_data_sample))
            random_seed = random_seed + 1

        return bootstrap_samples

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)

        bootstrap_samples = self.generate_bootstrap_sample(scenario, fold, self.num_base_learner)

        for index in range(self.num_base_learner):
            self.base_learners.append(copy.copy(self.base_learner))
            scenario.feature_data, scenario.performance_data, scenario.runstatus_data, scenario.feature_runstatus_data, scenario.feature_cost_data = bootstrap_samples[index]
            self.base_learners[index].fit(scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros((self.num_algorithms, 1))
        for model in self.base_learners:
            ranked_prediction = rankdata(model.predict(features_of_test_instance, instance_id)).reshape(
                self.num_algorithms, 1)
            predictions = predictions + (((self.num_algorithms + 1) - ranked_prediction) / self.num_algorithms)
        return 1 - predictions / self.num_base_learner

    def get_name(self):

        return "bagging_" + str(self.num_base_learner) + "_" + self.base_learner.get_name()
