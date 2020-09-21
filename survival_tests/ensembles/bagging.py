import logging

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import resample

from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario


class Bagging:

    def __init__(self, num_base_learner: int):
        self.logger = logging.getLogger("bagging")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.base_learners = list()
        self.num_base_learner = num_base_learner

    def generate_bootstrap_sample(self, scenario: ASlibScenario, fold: int, number_of_samples: int):
        samples = list()
        sample_size = len(scenario.instances)
        random_seed = (fold - 1) * number_of_samples

        for counter in range(number_of_samples):
            #TODO: Closer look at performance_data_all -> List with Datafame Object - Why? ---- feature_cost_data is also not a DataFrame
            feature_data_sample = scenario.feature_data.sample(sample_size, replace=True, random_state=random_seed)
            performance_data_sample = scenario.performance_data.sample(sample_size, replace=True, random_state=random_seed)
            performance_data_all_sample = scenario.performance_data_all[0].sample(sample_size, replace=True, random_state=random_seed)
            runstatus_data_sample = scenario.runstatus_data.sample(sample_size, replace=True, random_state=random_seed)
            #feature_cost_data_sample = scenario.feature_cost_data.sample(sample_size, replace=True, random_state=random_seed)
            feature_runstatus_data_sample = scenario.feature_runstatus_data.sample(sample_size, replace=True, random_state=random_seed)


            samples.append((feature_data_sample, performance_data_sample, performance_data_all_sample, runstatus_data_sample, feature_runstatus_data_sample))
            random_seed = random_seed + 1

        return samples

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)

        samples = self.generate_bootstrap_sample(scenario, fold, self.num_base_learner)

        for index in range(self.num_base_learner):
            self.base_learners.append(PerAlgorithmRegressor())
            scenario.feature_data, scenario.performance_data, scenario.performance_data_all[0], scenario.runstatus_data, scenario.feature_runstatus_data = samples[index]
            self.base_learners[index].fit(scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros((self.num_algorithms, 1))
        for model in self.base_learners:
            ranked_prediction = rankdata(model.predict(features_of_test_instance, instance_id)).reshape(
                self.num_algorithms, 1)
            predictions = predictions + (((self.num_algorithms + 1) - ranked_prediction) / self.num_algorithms)
        return 1 - predictions / self.num_base_learner

    # TODO: What does this method?
    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(performance_data, axis=0)) if num_instances > 0 else np.size(
            performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def get_name(self):

        return "bagging" + str(self.num_base_learner)
