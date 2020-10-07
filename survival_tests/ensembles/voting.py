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

    def __init__(self, agg_method='default', weighting=False):
        # logger
        self.logger = logging.getLogger("voting")
        self.logger.addHandler(logging.StreamHandler())

        # parameter
        self.agg_method = agg_method
        self.weighting = weighting

        # attributes
        self.trained_models = list()
        self.weights = list()
        self.metric = Par10Metric()
        self.num_algorithms = 0

    def create_base_learner(self):
        # clean up list and init base learners
        self.trained_models = list()

        self.trained_models.append(PerAlgorithmRegressor())
        #self.trained_models.append(SUNNY())
        #self.trained_models.append(ISAC())
        #self.trained_models.append(SATzilla11())
        self.trained_models.append(SurrogateAutoSurvivalForest())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()

        weights_denorm = list()

        # train base learner and calculate the weights
        for base_learner in self.trained_models:
            base_learner.fit(scenario, fold, amount_of_training_instances)
            if self.weighting:
                weights_denorm.append(self.base_learner_performance(scenario, amount_of_training_instances, base_learner))

        weights_denorm = [max(weights_denorm) / float(i) for i in weights_denorm]
        self.weights = [float(i) / sum(weights_denorm) for i in weights_denorm]

    def base_learner_performance(self, scenario: ASlibScenario, amount_of_training_instances: int, base_learner):
        # extract data from scenario
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        # performance_measure hold the PAR10 score for every instance
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
        if self.agg_method is 'ranking':
            return self.predict_with_ranking(features_of_test_instance, instance_id)

        if self.agg_method is 'stv':
            return self.predict_with_stv(features_of_test_instance, instance_id)

        # only using the prediction of the algorithm
        predictions = np.zeros(self.num_algorithms)
        for i, model in enumerate(self.trained_models):
            # get prediction of base learner and find prediction (lowest value)
            base_prediction = model.predict(features_of_test_instance, instance_id).reshape(self.num_algorithms)
            index_of_minimum = np.where(base_prediction == min(base_prediction))

            # add [1 * weight for base learner] to vote for the algorithm
            if self.weighting:
                predictions[index_of_minimum] = predictions[index_of_minimum] + self.weights[i]
            else:
                predictions[index_of_minimum] = predictions[index_of_minimum] + 1
        return 1 - predictions / sum(predictions)

    def predict_with_ranking(self, features_of_test_instance, instance_id: int):
        # use all predictions (ranking) of the base learner
        predictions = np.zeros((self.num_algorithms, 1))
        for i, model in enumerate(self.trained_models):
            # rank output from the base learner (best algorithm gets rank 1 - worst algorithm gets rank len(algorithms))
            if self.weighting:
                predictions = predictions + self.weights[i] * rankdata(
                    model.predict(features_of_test_instance, instance_id)).reshape(
                    self.num_algorithms, 1)
            else:
                predictions = predictions + rankdata(
                    model.predict(features_of_test_instance, instance_id)).reshape(
                    self.num_algorithms, 1)
        return predictions / sum(predictions)

    def predict_with_stv(self, features_of_test_instance, instance_id: int):
        print("Predict with stv")
        # quota to find the top three algorithms
        quota = int(self.num_algorithms * len(self.trained_models) / 2)

        predictions = np.zeros((len(self.trained_models), self.num_algorithms))
        for i, model in enumerate(self.trained_models):
            predictions[i] = rankdata(model.predict(features_of_test_instance, instance_id))#.reshape(self.num_algorithms, 1)

        final_prediction = None
        while final_prediction is None:
            colum_sum = np.count_nonzero(predictions == 1, axis=0)
            print(predictions)
            print(colum_sum)
            max_index = np.argmax(colum_sum)
            print(max_index)
            if colum_sum[max_index] > quota:
                print("final prediction")
                final_prediction = np.ones(self.num_algorithms)
                final_prediction[max_index] = 0
            else:
                min_index = self.min_without_zero(colum_sum)
                print("Min index:", min_index)
                for i in range(len(self.trained_models)):
                    print(np.argmin(predictions[i]) == min_index)
                    if np.argmin(predictions[i]) == min_index:
                        print("hier")
                        # TODO: change max value
                        predictions[i][min_index] = 100000
                        predictions[i] = rankdata(predictions[i])

        return final_prediction

    def min_without_zero(self, array):
        lowest_value = 100000
        i = 0
        for i, value in enumerate(array):
            if value < lowest_value and value != 0:
                lowest_value = value
                min_index = i
        return i

    def get_name(self):
        name = "voting" + "_" + self.agg_method
        if self.weighting:
            name = name + "_weighting"
        return name
