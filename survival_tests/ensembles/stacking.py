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
import pandas as pd


class Stacking:

    def __init__(self, meta_learner_type='per_algorithm_regressor', cross_validation=False, feature_selection=None, feature_importances=False, pre_computed=False):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.cross_validation = cross_validation
        self.feature_selection = feature_selection
        self.meta_learner_type = meta_learner_type
        self.feature_importances = feature_importances
        self.pre_computed = pre_computed


        # attributes
        self.meta_learner = None
        self.base_learners = list()
        self.num_algorithms = 0

    def create_base_learner(self):
        #self.base_learners.append(PerAlgorithmRegressor())
        self.base_learners.append(SUNNY())
        self.base_learners.append(ISAC())
        #self.base_learners.append(SATzilla11())
        #self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential'))
        #self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        #self.base_learners.append(MultiClassAlgorithmSelector())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setup
        if not self.pre_computed:
            self.create_base_learner()
        self.num_algorithms = len(scenario.algorithms)
        num_instances = len(scenario.instances)
        feature_data = scenario.feature_data.to_numpy()
        new_feature_data = np.zeros((num_instances, self.num_algorithms))

        instance_counter = 0

        # With or without 10-fold cross validation
        if self.cross_validation:
            for sub_fold in range(1, 11):
                test_scenario, training_scenario = self.split_scenario(scenario, sub_fold, num_instances)

                for learner_index, base_learner in enumerate(self.base_learners):
                    # train base learner
                    base_learner.fit(training_scenario, fold, amount_of_training_instances)

                    # create new feature data
                    for instance_number in range(instance_counter, instance_counter + len(test_scenario.instances)):
                        prediction = base_learner.predict(feature_data[instance_number], instance_number)
                        new_feature_data[instance_number][np.argmin(prediction)] = new_feature_data[instance_number][np.argmin(prediction)] + 1

                instance_counter = instance_counter + len(test_scenario.instances)
        else:
            for learner_index, base_learner in enumerate(self.base_learners):
                # train base learner
                if not self.pre_computed:
                    base_learner.fit(scenario, fold, amount_of_training_instances)

                # create new feature data
                num_iterations = len(scenario.instances) if amount_of_training_instances == -1 else amount_of_training_instances

                for instance_number in range(num_iterations):
                    prediction = base_learner.predict(feature_data[instance_number], instance_number)
                    new_feature_data[instance_number][np.argmin(prediction)] = new_feature_data[instance_number][np.argmin(prediction)] + 1
        # add predictions to the features of the instances
        new_feature_data = pd.DataFrame(new_feature_data, index=scenario.feature_data.index, columns=np.arange(self.num_algorithms))
        new_feature_data = pd.concat([scenario.feature_data, new_feature_data], axis=1, sort=False)
        scenario.feature_data = new_feature_data

        # meta learner training with or without feature selection
        if self.meta_learner_type == 'per_algorithm_regressor':
            self.meta_learner = PerAlgorithmRegressor(feature_selection=self.feature_selection, feature_importances=self.feature_importances)
        elif self.meta_learner_type == 'multiclass_algorithm_selector':
            self.meta_learner = MultiClassAlgorithmSelector(feature_selection=self.feature_selection)
        self.meta_learner.fit(scenario, fold, amount_of_training_instances)


    def predict(self, features_of_test_instance, instance_id: int):
        # get all predictions from the base learners
        new_feature_data = np.zeros(self.num_algorithms)

        for base_learner in self.base_learners:
            # create new feature data
            prediction = base_learner.predict(features_of_test_instance, instance_id)
            new_feature_data[np.argmin(prediction)] = new_feature_data[np.argmin(prediction)] + 1

        features_of_test_instance = np.concatenate((features_of_test_instance, new_feature_data), axis=0)

        # final prediction
        return self.meta_learner.predict(features_of_test_instance, instance_id)

    def get_name(self):
        name = "stacking_" + self.meta_learner_type
        if self.cross_validation:
            name = name + "_cross_validation"
        if self.feature_selection is not None:
            name = name + "_" + self.feature_selection

        return name
