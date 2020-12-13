import logging
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier

from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from ensembles.validation import split_scenario, get_confidence
import copy

from ensembles.validation import base_learner_performance


class StackingNew:

    def __init__(self, meta_learner_type='random_forest_classifier', cross_validation=False, feature_selection=None, base_learner=None, type='standard', pre_computed=False):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.cross_validation = cross_validation
        self.feature_selection = feature_selection
        self.meta_learner_type = meta_learner_type
        self.base_learner = base_learner
        self.type = type
        self.pre_computed = pre_computed

        # attributes
        self.meta_learner = None
        self.base_learners = list()
        self.num_algorithms = 0
        self.confidence = list()

    def create_base_learner(self):
        self.base_learners = list()

        if 1 in self.base_learner:
            self.base_learners.append(PerAlgorithmRegressor())
        if 2 in self.base_learner:
            self.base_learners.append(SUNNY())
        if 3 in self.base_learner:
            self.base_learners.append(ISAC())
        if 4 in self.base_learner:
            self.base_learners.append(SATzilla11())
        if 5 in self.base_learner:
            self.base_learners.append(SurrogateSurvivalForest(criterion='Expectation'))
        if 6 in self.base_learner:
            self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        if 7 in self.base_learner:
            self.base_learners.append(MultiClassAlgorithmSelector())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setting for pre computed base learner
        if not self.pre_computed:
            self.create_base_learner()
        
        # setup
        self.num_algorithms = len(scenario.algorithms)
        feature_data = scenario.feature_data.to_numpy()
        num_instances = len(feature_data)

        # create new feature data structure
        if self.type == 'standard' or self.type == 'confidence_prediction':
            x_train = np.zeros((num_instances, self.num_algorithms))
        elif self.type == 'full_prediction' or self.type == 'full_prediction_norm':
            x_train = [[] for x in range(num_instances)]
        else:
            sys.exit("Wrong prediction type!")

        # create the actual new feature data
        for i, base_learner in enumerate(self.base_learners):

            # train the base learner
            if not self.pre_computed:
                base_learner.fit(scenario, fold, amount_of_training_instances)

            # calculate the confidence of base learner i
            if self.type == 'confidence_prediction':
                self.append(get_confidence(scenario, amount_of_training_instances, base_learner))

            # predict with base learner i and create feature data
            for instance_number, x_test in enumerate(feature_data):
                algorithm_prediction = base_learner.predict(x_test, instance_number)

                # standard -> each base learner does one prediction -> [0 2 0 2 3]
                if self.type == 'standard':
                    algorithm_prediction = np.argmin(algorithm_prediction)
                    x_train[instance_number][algorithm_prediction] = x_train[instance_number][algorithm_prediction] + 1
                
                # confidence_prediction -> the confidences for the best prediction are added -> [0 1.3 4.2 5.3 0.1]
                elif self.type == 'confidence_prediction':
                    algorithm_prediction = np.argmin(algorithm_prediction)
                    x_train[instance_number][algorithm_prediction] = x_train[instance_number][algorithm_prediction] + self.confidence[i]
                
                # full_prediction -> the full prediction of all base learners is added to the feature data -> [1 1 1 1 0 34.5 245.3 3435. 253.3 253. ...]
                elif self.type == 'full_prediction':
                    x_train[instance_number].extend(algorithm_prediction.flatten())

                # full_prediction_norm -> the full normalzied prediction of all base learners is added to the feature data -> [1 1 1 1 0 0.2 0.5 ...]
                elif self.type == 'full_prediction_norm':
                    algorithm_prediction = algorithm_prediction / sum(algorithm_prediction)
                    x_train[instance_number].extend(algorithm_prediction.flatten())

        # setup the meta-learner
        if self.meta_learner_type == 'random_forest_classifier':
            self.meta_learner = RandomForestClassifier(n_jobs=1, n_estimators=100)
        else:
            sys.exit("Wrong meta learner type")
        self.meta_learner.set_params(random_state=fold)

        # train the meta learner
        y_train = list()
        for data in scenario.performance_data.to_numpy():
            y_train.append(np.argmin(data))
        self.meta_learner.fit(x_train, y_train)

    def predict(self, features_of_test_instance, instance_id: int):
        # create new feature data structure
        if self.type == 'standard' or self.type == 'confidence_prediction':
            new_feature_data = np.zeros(self.num_algorithms)
        elif self.type == 'full_prediction' or self.type == 'full_prediction_norm':
            new_feature_data = list()

        # create the actual new feature data -> see 'fit' method for more details
        for i, base_learner in enumerate(self.base_learners):
            algorithm_prediction = base_learner.predict(features_of_test_instance, instance_id)
            if self.type == 'standard':
                algorithm_prediction = np.argmin(algorithm_prediction)
                new_feature_data[algorithm_prediction] = new_feature_data[algorithm_prediction] + 1
            elif self.type == 'confidence_prediction':
                algorithm_prediction = np.argmin(algorithm_prediction)
                new_feature_data[algorithm_prediction] = new_feature_data[algorithm_prediction] + self.confidence[i]
            elif self.type == 'full_prediction':
                new_feature_data.extend(algorithm_prediction.flatten())
            elif self.type == 'full_prediction_norm':
                algorithm_prediction = algorithm_prediction / sum(algorithm_prediction)
                new_feature_data.extend(algorithm_prediction.flatten())

        new_feature_data = np.array(new_feature_data)
        features_of_test_instance = new_feature_data.reshape(1, -1)

        # predict with the meta-learner
        prediction = self.meta_learner.predict(features_of_test_instance)

        # create a final prediction where all algorithms get a 1 except the best one. This one gets a 0.
        final_prediction = np.ones(self.num_algorithms)
        final_prediction[prediction] = 0

        return final_prediction


    def get_name(self):
        name = "stacking_new_" + self.meta_learner_type + "_" + self.type
        if self.cross_validation:
            name = name + "_cross_validation"
        if self.feature_selection is not None:
            name = name + "_" + self.feature_selection

        return name
