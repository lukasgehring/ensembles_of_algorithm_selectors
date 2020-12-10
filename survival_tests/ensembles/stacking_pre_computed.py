from aslib_scenario.aslib_scenario import ASlibScenario

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import base_learner_performance, split_scenario
from ensembles.stacking_new import StackingNew
import dill
import numpy as np
import sys


class StackingPreComputed(StackingNew):

    def __init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, base_learner=None):
        StackingNew.__init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, base_learner=base_learner)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        if 1 in self.base_learner:
            self.base_learners.append(self.open_base_learner('per_algorithm_regressor', scenario.scenario, fold))
        if 2 in self.base_learner:
            self.base_learners.append(self.open_base_learner('sunny', scenario.scenario, fold))
        if 3 in self.base_learner:
            self.base_learners.append(self.open_base_learner('isac', scenario.scenario, fold))
        if 4 in self.base_learner:
            self.base_learners.append(self.open_base_learner('satzilla', scenario.scenario, fold))
        if 5 in self.base_learner:
            self.base_learners.append(self.open_base_learner('expectation', scenario.scenario, fold))
        if 6 in self.base_learner:
            self.base_learners.append(self.open_base_learner('par10', scenario.scenario, fold))
        if 7 in self.base_learner:
            self.base_learners.append(self.open_base_learner('multiclass', scenario.scenario, fold))

        self.num_algorithms = len(scenario.algorithms)
        feature_data = scenario.feature_data.to_numpy()
        num_instances = len(feature_data)
        new_feature_data = np.zeros((num_instances, self.num_algorithms))

        for i, base_learner in enumerate(self.base_learners):
            for instance_number, x_test in enumerate(feature_data):
                algorithm_prediction = np.argmin(base_learner.predict(x_test, instance_number))
                new_feature_data[instance_number][algorithm_prediction] = new_feature_data[instance_number][algorithm_prediction] + 1

        scenario.feature_data = pd.DataFrame(data=new_feature_data)

        for sub_fold in range(10):
            test_scenario, training_scenario = self.split_scenario(scenario, sub_fold + 1, num_instances)

            self.meta_learners.append(PerAlgorithmRegressor(feature_selection=self.feature_selection), None)
            self.meta_learners[sub_fold][0].fit(test_scenario, fold, amount_of_training_instances)
            self.meta_learners[sub_fold][1] = base_learner_performance(test_scenario, amount_of_training_instances, self.meta_learners[sub_fold][0])


    def predict(self, features_of_test_instance, instance_id: int):
        return StackingNew.predict(self, features_of_test_instance, instance_id)

    # open pre computed base learner
    def open_base_learner(self, base_learner_name, scenario_name, fold):
        file_name = 'pre_computed/' + base_learner_name + scenario_name + '_' + str(fold)
        with open(file_name, 'rb') as input:
            return dill.load(input)

    def get_name(self):
        return StackingNew.get_name(self) + '_pre'
