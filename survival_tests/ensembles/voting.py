import logging
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import rankdata

from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.isac import ISAC
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.utils import resample

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
        #self.trained_models.append(SurrogateAutoSurvivalForest())
        self.num_models = len(self.trained_models)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()
        self.weights = np.ones(self.num_models) / self.num_models

        for base_learner in self.trained_models:
            base_learner.fit(scenario, fold, amount_of_training_instances)

        bounds = [(0, 2), (0, 2), (0, 2), (0, 2)]
        result = differential_evolution(self.validation, bounds, args=(scenario, fold, amount_of_training_instances), seed=fold, workers=4, maxiter=3, disp=True)
        self.weights = result.x


    def validation(self, weights, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Call validate with weight", weights)
        # if scenario.feature_cost_data is not None:
        #     feature_data, performance_data, feature_cost_data = self._resample_instances(
        #         scenario.feature_data, scenario.performance_data, scenario.feature_cost_data, amount_of_training_instances, random_state=fold)
        # else:
        #     feature_data, performance_data = self._resample_instances(
        #         scenario.feature_data, scenario.performance_data, scenario.feature_cost_data,
        #         amount_of_training_instances, random_state=fold)
        #     feature_cost_data = None

        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        if amount_of_training_instances is -1:
            amount_of_training_instances = len(feature_data)

        par10 = 0
        for instance_id in range(amount_of_training_instances):
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if scenario.feature_cost_data is not None:
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            self.weights = weights

            predicted_scores = self.predict(x_test, instance_id)
            par10 = par10 + self.metric.evaluate(y_test, predicted_scores, accumulated_feature_time,
                                                 scenario.algorithm_cutoff_time)
        return par10 / amount_of_training_instances

    # def create_split(self, scenario: ASlibScenario, fold: int):
    #     split_size = int(len(scenario.instances) / 10)
    #
    #     feature_data = scenario.feature_data.to_numpy()
    #     performance_data = scenario.performance_data.to_numpy()
    #
    #     if fold is 10:
    #         training_split_feature = feature_data[:split_size * (fold - 1):]
    #         test_split_feature = feature_data[split_size * (fold - 1):]
    #
    #         training_split_performance = performance_data[:split_size * (fold - 1):]
    #         test_split_performance = performance_data[split_size * (fold - 1):]
    #     else:
    #         training_split_feature = feature_data[:split_size * (fold - 1)]
    #         training_split_feature.extend(feature_data[split_size * fold:])
    #         test_split_feature = feature_data[split_size * (fold - 1):split_size * fold]
    #
    #         training_split_performance = performance_data[:split_size * (fold - 1)]
    #         training_split_performance.extend(performance_data[split_size * fold:])
    #         test_split_performance = performance_data[split_size * (fold - 1):split_size * fold]
    #     return training_split_feature, test_split_feature, training_split_performance, test_split_performance

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros((self.num_algorithms, 1))
        for i, model in enumerate(self.trained_models):
            predictions = predictions + self.weights[i] * rankdata(model.predict(features_of_test_instance, instance_id)).reshape(
                self.num_algorithms, 1)
        return predictions / self.num_models

    def _resample_instances(self, feature_data, performance_data, feature_cost_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(
            performance_data, axis=0)) if num_instances > 0 else np.size(performance_data, axis=0)
        if feature_cost_data is not None:
            return resample(feature_data, performance_data, feature_cost_data, n_samples=num_instances, random_state=random_state)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def get_name(self):
        return "voting"
