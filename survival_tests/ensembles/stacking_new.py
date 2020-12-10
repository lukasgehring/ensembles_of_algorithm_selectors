import logging
import numpy as np
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
import copy


class StackingNew:

    def __init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, base_learner=None):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.cross_validation = cross_validation
        self.feature_selection = feature_selection
        self.meta_learner_type = meta_learner_type
        self.base_learner = base_learner


        # attributes
        self.meta_learners = list()
        self.base_learners = list()
        self.num_algorithms = 0

    def create_base_learner(self):
        self.base_learners = list()

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

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setup
        self.create_base_learner()
        self.num_algorithms = len(scenario.algorithms)
        feature_data = scenario.feature_data.to_numpy()
        num_instances = len(feature_data)
        new_feature_data = np.zeros((num_instances, self.num_algorithms))

        for i, base_learner in enumerate(self.base_learners):
            base_learners.fit(scenario, fold, amount_of_training_instances)
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
        # get all predictions from the base learners
        new_feature_data = np.zeros(self.num_algorithms)

        for base_learner in self.base_learners:
            # create new feature data
            algorithm_prediction = np.argmin(base_learner.predict(features_of_test_instance, instance_id))
            new_feature_data[algorithm_prediction] = new_feature_data[algorithm_prediction] + 1

        features_of_test_instance = new_feature_data

        final_prediction = np.zeros(self.num_algorithms)

        # final prediction
        for meta_learner, performance in self.meta_learners:
            algorithm_prediciton = np.argmin(meta_learner.predict(features_of_test_instance, instance_id))
            final_prediction[algorithm_prediciton] = final_prediction[algorithm_prediciton] + 1 / performance

        return final_prediction

    def get_name(self):
        name = "stacking_new_" + self.meta_learner_type
        if self.cross_validation:
            name = name + "_cross_validation"
        if self.feature_selection is not None:
            name = name + "_" + self.feature_selection

        return name
