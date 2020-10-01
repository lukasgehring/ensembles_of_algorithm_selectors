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

        self.trained_models.append(PerAlgorithmRegressor())
        self.trained_models.append(SUNNY())
        self.trained_models.append(ISAC())
        self.trained_models.append(SATzilla11())
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
<<<<<<< HEAD
        return par10 / amount_of_training_instances

    def differential_evolution(self, scenario: ASlibScenario, fold:int, amount_of_training_instances):
        # at least 4
        num_individuals = 10
        min_bound = 0
        max_bound = 5
        diff = max_bound - min_bound

        best_score = 1000000
        optimal_weights = self.weights

        #TODO: Seed
        np.random.seed(fold)
        population = np.random.rand(num_individuals, self.num_models)
        population_fit_to_bounds = min_bound + population * diff
        scaling_factor = 0.8
        crossover = 0.5
        generation = 0
        while generation < 20:
            print("Generation", generation, "started its training now")
            for i in range(num_individuals):
                selections = [j for j in range(num_individuals) if j != i]
                random_selections = np.random.choice(selections, 3,
                                                     replace=False)
                a, b, c = population[random_selections]
                mutant = a + scaling_factor * (b - c)
                j_random = np.random.rand(1, 1)
                tail = list()
                for j in range(self.num_models):
                    rand = np.random.rand(1, 1)
                    if rand < crossover or rand == j_random:
                        tail.append(mutant[j])
                    else:
                        tail.append(population[i][j])
                tail_score = self.validation(scenario, amount_of_training_instances, tail)
                pop_score = self.validation(scenario, amount_of_training_instances, population[i])
                if tail_score < pop_score:
                    population[i] = tail
                    if tail_score < best_score:
                        best_score = tail_score
                        optimal_weights = tail
                else:
                    if pop_score < best_score:
                        best_score = pop_score
                        optimal_weights = pop_score[i]

            generation = generation + 1

        return optimal_weights
=======
        return performance_measure / amount_of_training_instances
>>>>>>> 708f48a3bbeef6ff3f3e0ec80b56b065e8d4b84b

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
