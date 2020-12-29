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
import sys

from ensembles.validation import split_scenario
from pre_compute.pickle_loader import load_pickle


class StackingH2O:

    def __init__(self, meta_learner_type='per_algorithm_regressor', pre_computed=False):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.meta_learner_type = meta_learner_type
        self.pre_computed = pre_computed


        # attributes
        self.meta_learner = None
        self.base_learners = list()
        self.num_algorithms = 0
        self.scenario_name = ''
        self.fold = 0
        self.predictions = list()

    def create_base_learner(self):
        self.base_learners.append(PerAlgorithmRegressor())
        self.base_learners.append(SUNNY())
        #self.base_learners.append(ISAC())
        #self.base_learners.append(SATzilla11())
        #self.base_learners.append(SurrogateSurvivalForest(criterion='Exponential'))
        #self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        self.base_learners.append(MultiClassAlgorithmSelector())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.create_base_learner()
        self.scenario_name = scenario.scenario
        self.fold = fold
        self.num_algorithms = len(scenario.algorithms)
        num_instances = len(scenario.instances)
        feature_data = scenario.feature_data.to_numpy()
        new_feature_data = np.zeros((num_instances, self.num_algorithms * len(self.base_learners)))

        for learner_index, base_learner in enumerate(self.base_learners):

            instance_counter = 0

            predictions = np.zeros((num_instances, self.num_algorithms))

            if self.pre_computed:
                predictions = load_pickle(filename='predictions/cross_validation_' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold))
            else:
                for sub_fold in range(1, 11):
                    test_scenario, training_scenario = split_scenario(scenario, sub_fold, num_instances)

                    # train base learner
                    base_learner.fit(training_scenario, fold, amount_of_training_instances)

                    # create new feature data
                    for instance_number in range(instance_counter, instance_counter + len(test_scenario.instances)):
                        prediction = base_learner.predict(feature_data[instance_number], instance_number)
                        predictions[instance_number] = prediction

                    instance_counter = instance_counter + len(test_scenario.instances)

            for i in range(num_instances):
                for alo_num in range(self.num_algorithms):
                    new_feature_data[i][alo_num + self.num_algorithms * learner_index] = predictions[i][alo_num]

        # add predictions to the features of the instances
        new_feature_data = pd.DataFrame(new_feature_data, index=scenario.feature_data.index, columns=np.arange(self.num_algorithms * len(self.base_learners)))
        new_feature_data = pd.concat([scenario.feature_data, new_feature_data], axis=1, sort=False)
        scenario.feature_data = new_feature_data

        # meta learner training with or without feature selection
        if self.meta_learner_type == 'per_algorithm_regressor':
            self.meta_learner = PerAlgorithmRegressor()
        elif self.meta_learner_type == 'SUNNY':
            self.meta_learner = SUNNY()
        self.meta_learner.fit(scenario, fold, amount_of_training_instances)

        if self.pre_computed:
            for base_learner in self.base_learners:
                self.predictions.append(load_pickle(filename='predictions/' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold)))





    def predict(self, features_of_test_instance, instance_id: int):
        # get all predictions from the base learners
        new_feature_data = np.zeros(self.num_algorithms * len(self.base_learners))

        for learner_index, base_learner in enumerate(self.base_learners):
            # create new feature data
            if self.pre_computed:
                prediction = self.predictions[learner_index]
            else:
                prediction = base_learner.predict(features_of_test_instance, instance_id)

            for alo_num in range(self.num_algorithms):
                new_feature_data[alo_num + self.num_algorithms * learner_index] = prediction[str(features_of_test_instance)][alo_num]

        features_of_test_instance = np.concatenate((features_of_test_instance, new_feature_data), axis=0)

        # final prediction
        return self.meta_learner.predict(features_of_test_instance, instance_id)

    def get_name(self):
        name = "stacking_" + self.meta_learner_type

        return name
