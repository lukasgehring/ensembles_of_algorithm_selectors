import logging
import numpy as np
from scipy.stats import rankdata

from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.isac import ISAC
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from aslib_scenario.aslib_scenario import ASlibScenario

from par_10_metric import Par10Metric


class Voting:

    def __init__(self):
        self.logger = logging.getLogger("voting")
        self.logger.addHandler(logging.StreamHandler())
        self.trained_models = list()
        self.num_algorithms = 0
        self.num_models = 0
        self.metric = Par10Metric()
        self.weights = list()

    def create_base_learner(self):
        self.trained_models = list()
        self.trained_models.append(PerAlgorithmRegressor())
        self.trained_models.append(SUNNY())
        self.trained_models.append(ISAC())
        self.trained_models.append(SATzilla11())
        self.trained_models.append(SurrogateAutoSurvivalForest())
        self.num_models = len(self.trained_models)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()

        weights_denorm = list()

        for base_learner in self.trained_models:
            base_learner.fit(scenario, fold, amount_of_training_instances)
            weights_denorm.append(self.base_learner_performance(scenario, amount_of_training_instances, base_learner))

        weights_denorm = [max(weights_denorm) / float(i) for i in weights_denorm]
        self.weights = [float(i) / sum(weights_denorm) for i in weights_denorm]

    def base_learner_performance(self, scenario: ASlibScenario, amount_of_training_instances: int, base_learner):
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        performance_measure = 0
        for instance_id in range(amount_of_training_instances):
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if scenario.feature_cost_data is not None:
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            predicted_scores = base_learner.predict(x_test, instance_id)
            performance_measure = performance_measure + self.metric.evaluate(y_test, predicted_scores, accumulated_feature_time,
                                                 scenario.algorithm_cutoff_time)
        return performance_measure / amount_of_training_instances

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros((self.num_algorithms, 1))
        for i, model in enumerate(self.trained_models):
            predictions = predictions + self.weights[i] * rankdata(model.predict(features_of_test_instance, instance_id)).reshape(
                self.num_algorithms, 1)
        return predictions / self.num_models

    def get_name(self):
        return "voting"
