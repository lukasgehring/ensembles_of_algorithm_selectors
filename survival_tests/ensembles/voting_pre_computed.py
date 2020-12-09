from aslib_scenario.aslib_scenario import ASlibScenario

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import base_learner_performance, split_scenario
from ensembles.voting import Voting
import dill
import numpy as np
import sys


class VotingPreComputed(Voting):

    def __init__(self, ranking=False, weighting=False, cross_validation=False, base_learner=None,
                 rank_method='average'):
        Voting.__init__(self, base_learner=base_learner, ranking=ranking, weighting=weighting, cross_validation=cross_validation)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)

        if self.cross_validation:

            if 1 in self.base_learner:
                self.trained_models.append(PerAlgorithmRegressor())
            if 2 in self.base_learner:
                self.trained_models.append(SUNNY())
            if 3 in self.base_learner:
                self.trained_models.append(ISAC())
            if 4 in self.base_learner:
                self.trained_models.append(SATzilla11())
            if 5 in self.base_learner:
                self.trained_models.append(SurrogateSurvivalForest(criterion='Expectation'))
            if 6 in self.base_learner:
                self.trained_models.append(SurrogateSurvivalForest(criterion='PAR10'))
            if 7 in self.base_learner:
                self.trained_models.append(MultiClassAlgorithmSelector())

            weights_denorm = np.zeros(len(self.trained_models))
            num_instances = len(scenario.instances)

            # cross validation for the weight
            for sub_fold in range(1, 11):
                test_scenario, training_scenario = split_scenario(scenario, sub_fold, num_instances)
                # train base learner and calculate the weights
                for i, base_learner in enumerate(self.trained_models):
                    base_learner.fit(training_scenario, fold, amount_of_training_instances)
                    weights_denorm[i] = weights_denorm[i] + base_learner_performance(test_scenario,
                                                                                     amount_of_training_instances,
                                                                                     base_learner)

        self.trained_models = list()
        if len(self.trained_models) != 0:
            sys.exit("List of base learner not empty!")

        if 1 in self.base_learner:
            self.trained_models.append(self.open_base_learner('per_algorithm_regressor', scenario.scenario, fold))
        if 2 in self.base_learner:
            self.trained_models.append(self.open_base_learner('sunny', scenario.scenario, fold))
        if 3 in self.base_learner:
            self.trained_models.append(self.open_base_learner('isac', scenario.scenario, fold))
        if 4 in self.base_learner:
            self.trained_models.append(self.open_base_learner('satzilla', scenario.scenario, fold))
        if 5 in self.base_learner:
            self.trained_models.append(self.open_base_learner('expectation', scenario.scenario, fold))
        if 6 in self.base_learner:
            self.trained_models.append(self.open_base_learner('par10', scenario.scenario, fold))
        if 7 in self.base_learner:
            self.trained_models.append(self.open_base_learner('multiclass', scenario.scenario, fold))

        if self.cross_validation == False:
            weights_denorm = list()

            # train base learner and calculate the weights
            if self.weighting:
                for base_learner in self.trained_models:
                    weights_denorm.append(base_learner_performance(scenario, amount_of_training_instances, base_learner))

        # Turn around values (lowest (best) gets highest weight) and normalize
        weights_denorm = [max(weights_denorm) / float(i + 1) for i in weights_denorm]
        self.weights = [float(i) / max(weights_denorm) for i in weights_denorm]

    def predict(self, features_of_test_instance, instance_id: int):
        return Voting.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return Voting.get_name(self) + '_pre'
