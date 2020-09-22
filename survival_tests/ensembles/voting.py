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
        self.weights = np.ones(self.num_models) / self.num_models

        for base_learner in self.trained_models:
            base_learner.fit(scenario, fold, amount_of_training_instances)

        self.weights = self.differential_evolution(scenario, amount_of_training_instances)

    def validation(self, scenario: ASlibScenario, amount_of_training_instances: int, weights):
        # TODO: Add 10 fold cross validation?
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

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

    def differential_evolution(self, scenario: ASlibScenario, amount_of_training_instances):
        # at least 4
        num_individuals = 10
        min_bound = 0
        max_bound = 5
        diff = max_bound - min_bound

        best_score = 1000000
        optimal_weights = self.weights

        #TODO: Seed
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

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros((self.num_algorithms, 1))
        for i, model in enumerate(self.trained_models):
            predictions = predictions + self.weights[i] * rankdata(model.predict(features_of_test_instance, instance_id)).reshape(
                self.num_algorithms, 1)
        return predictions / self.num_models

    def get_name(self):
        return "voting"
