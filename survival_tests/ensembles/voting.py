import logging
import pandas as pd
import numpy as np
from matplotlib.pyplot import sci
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.base import clone
from scipy.stats import rankdata

from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla07 import SATzilla07
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from baselines.utils import impute_censored, distr_func
from aslib_scenario.aslib_scenario import ASlibScenario


class Voting:

    def __init__(self):
        self.logger = logging.getLogger("voting")
        self.logger.addHandler(logging.StreamHandler())
        self.trained_models = list()
        self.num_algorithms = 0
        self.num_models = 0

    def create_base_learner(self):
        self.trained_models = list()
        self.trained_models.append(PerAlgorithmRegressor())
        self.trained_models.append(SUNNY())
        self.trained_models.append(ISAC())
        #self.trained_models.append(SATzilla07())
        self.trained_models.append(SATzilla11())
        self.trained_models.append(SurrogateAutoSurvivalForest())
        self.num_models = len(self.trained_models)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()

        for base_learner in self.trained_models:
            base_learner.fit(scenario, fold, amount_of_training_instances)

    def predict(self, features_of_test_instance, instance_id: int):
        predictions = np.zeros((self.num_algorithms, 1))
        for model in self.trained_models:
            predictions = predictions + rankdata(model.predict(features_of_test_instance, instance_id)).reshape(self.num_algorithms, 1)
        return predictions / self.num_models

    def get_name(self):
        return "voting"
