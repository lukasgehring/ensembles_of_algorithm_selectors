import copy
import logging
import numpy as np
from scipy.stats import rankdata
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from aslib_scenario.aslib_scenario import ASlibScenario
from par_10_metric import Par10Metric


class Voting:

    def __init__(self, ranking=False, weighting=False):
        # logger
        self.logger = logging.getLogger("voting")
        self.logger.addHandler(logging.StreamHandler())

        # parameter
        self.ranking = ranking
        self.weighting = weighting

        # attributes
        self.trained_models = list()
        self.weights = list()
        self.metric = Par10Metric()
        self.num_algorithms = 0

    def create_base_learner(self):
        # clean up list and init base learners
        self.trained_models = list()

        #self.trained_models.append(PerAlgorithmRegressor())
        self.trained_models.append(SUNNY())
        self.trained_models.append(ISAC())
        #self.trained_models.append(SATzilla11())
        #self.trained_models.append(SurrogateSurvivalForest(criterion='Expectation'))
        #self.trained_models.append(SurrogateSurvivalForest(criterion='PAR10'))

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()

        num_instances = len(scenario.instances)
        np.random.seed(seed=fold)
        index_array = np.random.choice(num_instances, num_instances, replace=False)

        weights_denorm = np.zeros(len(self.trained_models))

        for sub_fold in range(1,11):
            print("Sub-fold", sub_fold)
            test_scenario, training_scenario = self.split_scenario(scenario, sub_fold, num_instances, index_array)
            # train base learner and calculate the weights
            for i, base_learner in enumerate(self.trained_models):
                print("Base learner", i)
                base_learner.fit(training_scenario, fold, amount_of_training_instances)
                weights_denorm[i] = weights_denorm[i] + self.base_learner_performance(test_scenario, base_learner)
            print("weights", weights_denorm)

        base_learner.fit(scenario, fold, amount_of_training_instances)

        weights_denorm = [max(weights_denorm) / float(i) for i in weights_denorm]
        self.weights = [float(i) / sum(weights_denorm) for i in weights_denorm]
        print(self.weights)

    def base_learner_performance(self, scenario: ASlibScenario, base_learner):
        # extract data from scenario
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        # performance_measure hold the PAR10 score for every instance
        performance_measure = 0
        for instance_id in range(len(scenario.instances)):
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if scenario.feature_cost_data is not None:
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            predicted_scores = base_learner.predict(x_test, instance_id)
            performance_measure = performance_measure + self.metric.evaluate(y_test, predicted_scores, accumulated_feature_time,
                                                 scenario.algorithm_cutoff_time)
        return performance_measure / len(scenario.instances)

    def split_scenario(self, scenario: ASlibScenario, sub_fold: int, num_instances: int, index):
        fold_len = int(num_instances / 10)
        instances = scenario.instances
        if sub_fold < 10:
            test_insts = instances[(sub_fold - 1) * fold_len:sub_fold * fold_len]
            training_insts = instances[:(sub_fold - 1) * fold_len]
            training_insts = np.append(training_insts, instances[sub_fold * fold_len:])
        else:
            test_insts = instances[(sub_fold - 1) * fold_len:]
            training_insts = instances[:(sub_fold - 1) * fold_len]

        test = copy.copy(scenario)
        training = copy.copy(scenario)

        # feature_data
        test.feature_data = test.feature_data.drop(training_insts).sort_index()
        training.feature_data = training.feature_data.drop(
            test_insts).sort_index()
        # performance_data
        test.performance_data = test.performance_data.drop(
            training_insts).sort_index()
        training.performance_data = training.performance_data.drop(
            test_insts).sort_index()
        # runstatus_data
        test.runstatus_data = test.runstatus_data.drop(
            training_insts).sort_index()
        training.runstatus_data = training.runstatus_data.drop(
            test_insts).sort_index()
        # self.feature_runstatus_data
        test.feature_runstatus_data = test.feature_runstatus_data.drop(
            training_insts).sort_index()
        training.feature_runstatus_data = training.feature_runstatus_data.drop(
            test_insts).sort_index()
        # feature_cost_data
        if scenario.feature_cost_data is not None:
            test.feature_cost_data = test.feature_cost_data.drop(
                training_insts).sort_index()
            training.feature_cost_data = training.feature_cost_data.drop(
                test_insts).sort_index()
        # ground_truth_data
        if scenario.ground_truth_data is not None:
            test.ground_truth_data = test.ground_truth_data.drop(
                training_insts).sort_index()
            training.ground_truth_data = training.ground_truth_data.drop(
                test_insts).sort_index()
        test.cv_data = None
        training.cv_data = None

        test.instances = test_insts
        training.instances = training_insts

        scenario.used_feature_groups = None

        return test, training

    def predict(self, features_of_test_instance, instance_id: int):
        if self.ranking:
            return self.predict_with_ranking(features_of_test_instance, instance_id)

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

    def get_name(self):
        name = "voting"
        if self.ranking:
            name = name + "_ranking"
        if self.weighting:
            name = name + "_weighting"
        return name
