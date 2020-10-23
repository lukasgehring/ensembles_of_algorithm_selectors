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


class Stacking:

    def __init__(self, cross_validation=False, feature_selection=None):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.cross_validation = cross_validation
        self.feature_selection = feature_selection

        # attributes
        self.meta_learner = None
        self.base_learners = list()
        self.num_algorithms = 0

    def create_base_learner(self):
        self.base_learners.append(PerAlgorithmRegressor())
        self.base_learners.append(SUNNY())
        self.base_learners.append(ISAC())
        self.base_learners.append(SATzilla11())
        self.base_learners.append(MultiClassAlgorithmSelector())
        self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential'))
        self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setup
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
                base_learner.fit(scenario, fold, amount_of_training_instances)

                # create new feature data
                for instance_number in range(amount_of_training_instances):
                    prediction = base_learner.predict(feature_data[instance_number], instance_number)
                    new_feature_data[instance_number][np.argmin(prediction)] = new_feature_data[instance_number][np.argmin(prediction)] + 1

        # add predictions to the features of the instances
        new_feature_data = np.concatenate((feature_data, new_feature_data), axis=1)
        scenario.feature_data = new_feature_data

        # meta learner training with or without feature selection
        self.meta_learner = PerAlgorithmRegressor(feature_selection=self.feature_selection)
        self.meta_learner.fit(scenario, fold, amount_of_training_instances)

    def split_scenario(self, scenario: ASlibScenario, sub_fold: int, num_instances: int):
        fold_len = int(num_instances / 10)

        # TODO: Do I need this? ----------------
        instances = scenario.instances
        if sub_fold < 10:
            test_insts = instances[(sub_fold - 1) * fold_len:sub_fold * fold_len]
            training_insts = instances[:(sub_fold - 1) * fold_len]
            training_insts = np.append(training_insts, instances[sub_fold * fold_len:])
        else:
            test_insts = instances[(sub_fold - 1) * fold_len:]
            training_insts = instances[:(sub_fold - 1) * fold_len]
        # TODO: --------------------------------

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
        name = "stacking"
        if self.cross_validation:
            name = name + "_cross_validation"
        if self.feature_selection is not None:
            name = name + "_" + self.feature_selection
        return name
