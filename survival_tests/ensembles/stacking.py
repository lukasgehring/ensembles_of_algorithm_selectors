import logging
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
import pandas as pd
from pre_compute.pickle_loader import load_pickle


class Stacking:

    def __init__(self, base_learner=None, meta_learner_type='per_algorithm_regressor', pre_computed=False, meta_learner_input='full', new_feature_type='full'):
        """
        Stacking Ensemble

        Params:
            - base_learner: List with all base learner numbers for the ensemble. Numbers can go from 1-7
            - meta_learner_type: Type of meta-learner
            - pre_computed: True, if predictions of the base learner are pre computed
            - meta_learner_input [full, predictions_only]: full = predictions + instance features,
                                                           predictions_only = predictions
            - new_feature_type [full, small]: full = all predictions of a base learner,
                                              small = only the best algorithm is selected
        """

        # logger
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.meta_learner_type = meta_learner_type
        self.pre_computed = pre_computed
        self.base_learner_type = base_learner
        self.meta_learner_input = meta_learner_input

        # attributes
        self.meta_learner = None
        self.base_learners = list()

        self.num_algorithms = 0
        self.scenario_name = ''
        self.fold = 0

        self.predictions = list()

        self.algorithm_selection_algorithm = False
        self.pipe = None

    def create_base_learner(self):
        self.base_learners = list()
        if 1 in self.base_learner_type:
            self.base_learners.append(PerAlgorithmRegressor())
        if 2 in self.base_learner_type:
            self.base_learners.append(SUNNY())
        if 3 in self.base_learner_type:
            self.base_learners.append(ISAC())
        if 4 in self.base_learner_type:
            self.base_learners.append(SATzilla11())
        if 5 in self.base_learner_type:
            self.base_learners.append(SurrogateSurvivalForest(criterion='Expectation'))
        if 6 in self.base_learner_type:
            self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        if 7 in self.base_learner_type:
            self.base_learners.append(MultiClassAlgorithmSelector())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):

        # setup the ensemble
        self.create_base_learner()
        self.scenario_name = scenario.scenario
        self.fold = fold
        self.num_algorithms = len(scenario.algorithms)

        num_instances = len(scenario.instances)
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()

        # new features in matrix [instances x predictions]
        if self.new_feature_type == 'full':
            new_feature_data = np.zeros((num_instances, self.num_algorithms * len(self.base_learners)))

        elif self.new_feature_type == 'small':
            new_feature_data = np.zeros((num_instances, len(self.base_learners)))

        # if predictions are precomputed
        if self.pre_computed:
            for base_learner in self.base_learners:
                self.predictions.append(load_pickle(filename='predictions/' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold)))

        # create new features for every base learner on each instance
        for learner_index, base_learner in enumerate(self.base_learners):

            if self.pre_computed:
                predictions = load_pickle(filename='predictions/full_trainingdata_' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold))

            else:
                base_learner.fit(scenario, fold, amount_of_training_instances)

                predictions = np.zeros((len(scenario.instances), self.num_algorithms))

                for instance_id, instance_feature in enumerate(feature_data):
                    predictions[instance_id] = base_learner.predict(instance_feature, instance_id)

            # insert predictions to new feature data matrix
            for i in range(num_instances):
                for alo_num in range(self.num_algorithms):

                    if self.new_feature_type == 'full':
                        new_feature_data[i][alo_num + self.num_algorithms * learner_index] = predictions[i][alo_num]

                    elif self.new_feature_type == 'small':
                        new_feature_data[i][learner_index] = np.argmin(predictions[i][alo_num])

        # add predictions to the features of the instances
        if self.new_feature_type == 'full':
            new_columns = np.arange(self.num_algorithms * len(self.base_learners))

        elif self.new_feature_type == 'small':
            new_columns = np.arange(len(self.base_learners))

        new_feature_data = pd.DataFrame(new_feature_data, index=scenario.feature_data.index, columns=new_columns)

        if self.meta_learner_input == 'full':
            new_feature_data = pd.concat([scenario.feature_data, new_feature_data], axis=1, sort=False)

        elif self.meta_learner_input == 'predictions_only':
            pass

        else:
            sys.exit('Wrong meta learner input type option')

        scenario.feature_data = new_feature_data

        # meta learner selection
        if self.meta_learner_type == 'per_algorithm_regressor':
            self.meta_learner = PerAlgorithmRegressor()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'SUNNY':
            self.meta_learner = SUNNY()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'ISAC':
            self.meta_learner = ISAC()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'SATzilla-11':
            self.meta_learner = SATzilla11()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'multiclass':
            self.meta_learner = MultiClassAlgorithmSelector()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'Expectation':
            self.meta_learner = SurrogateSurvivalForest(criterion='Expectation')
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'PAR10':
            self.meta_learner = SurrogateSurvivalForest(criterion='PAR10')
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'RandomForest':
            self.meta_learner = RandomForestClassifier(random_state=fold)
        elif self.meta_learner_type == 'SVM':
            self.meta_learner = LinearSVC(random_state=fold, max_iter=10000)

        # fit meta learner
        if self.algorithm_selection_algorithm:
            self.meta_learner.fit(scenario, fold, amount_of_training_instances)
        else:
            label_performance_data = [np.argmin(x) for x in performance_data]

            self.pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler())])
            x_train = self.pipe.fit_transform(scenario.feature_data.to_numpy(), label_performance_data)

            self.meta_learner.fit(x_train, label_performance_data)

    def predict(self, features_of_test_instance, instance_id: int):

        # get all predictions from the base learners
        if self.new_feature_type == 'full':
            new_feature_data = np.zeros(self.num_algorithms * len(self.base_learners))
        elif self.new_feature_type == 'small':
            new_feature_data = np.zeros(len(self.base_learners))
        for learner_index, base_learner in enumerate(self.base_learners):
            # create new feature data
            if self.pre_computed:
                prediction = self.predictions[learner_index][str(features_of_test_instance)]
            else:
                prediction = base_learner.predict(features_of_test_instance, instance_id).flatten()

            for alo_num in range(self.num_algorithms):
                if self.new_feature_type == 'full':
                    new_feature_data[alo_num + self.num_algorithms * learner_index] = prediction[alo_num]
                elif self.new_feature_type == 'small':
                    new_feature_data[learner_index] = np.argmin(prediction[alo_num])

        # concatenate original feature with new feature
        if self.meta_learner_input == 'full':
            features_of_test_instance = np.concatenate((features_of_test_instance, new_feature_data), axis=0)
        else:
            features_of_test_instance = new_feature_data

        # final prediction with the meta learner
        if self.algorithm_selection_algorithm:
            return self.meta_learner.predict(features_of_test_instance, instance_id)
        else:
            x_test = self.pipe.transform(features_of_test_instance.reshape(1, -1))
            final_prediction = np.ones(self.num_algorithms)
            final_prediction[self.meta_learner.predict(x_test)] = 0
            return final_prediction

    def get_name(self):
        name = "stacking" + "_" + str(self.base_learner_type).replace('[', '').replace(']', '').replace(', ', '_') + self.meta_learner_type
        return name
