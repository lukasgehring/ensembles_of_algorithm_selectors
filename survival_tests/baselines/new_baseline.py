from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import numpy as np
import pandas as pd
from scipy.spatial import distance
import sys


class NewBaseline:

    def __init__(self, distance_function='euclidean'):
        self.cluster_centroids = None
        self.imputer = None
        self.standard_scaler = None
        self.distance_function = distance_function

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        X_train, y_train = self.get_x_y(scenario, amount_of_training_instances, fold)

        # impute missing values
        self.imputer = SimpleImputer()
        X_train = self.imputer.fit_transform(X_train)

        # standardize feature values
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        # create clusters
        solver_cluster = [[] for _ in scenario.algorithms]
        self.cluster_centroids = []

        for features, performances in zip(X_train, y_train):
            best_solver = np.argmin(performances)
            if performances[best_solver] <= scenario.algorithm_cutoff_time:
                solver_cluster[best_solver].append(features)

        # determine centroids
        for s in solver_cluster:
            if len(s) > 0:
                self.cluster_centroids.append(np.average(s, axis=0))
            else:
                self.cluster_centroids.append(None)

    def predict(self, features_of_test_instance, instance_id: int):
        X_test = np.reshape(features_of_test_instance, (1, len(features_of_test_instance)))
        X_test = self.imputer.transform(X_test)
        X_test = self.scaler.transform(X_test)

        dist = float('inf')
        best_solver = 0
        for i, centroid in enumerate(self.cluster_centroids):
            if centroid is None:
                continue

            if self.distance_function == 'euclidean':
                cur_dist = distance.euclidean(centroid, X_test)
            elif self.distance_function == 'cosine':
                cur_dist = distance.cosine(centroid, X_test)
            else:
                sys.exit("Wrong distance function")

            if cur_dist < dist:
                dist = cur_dist
                best_solver = i

        prediction = np.ones(len(self.cluster_centroids))
        prediction[best_solver] = 0
        return prediction

    def get_x_y(self, scenario: ASlibScenario, num_requested_instances: int, fold: int):
        amount_of_training_instances = min(num_requested_instances,
                                           len(scenario.instances)) if num_requested_instances > 0 else len(
            scenario.instances)
        resampled_scenario_feature_data, resampled_scenario_performances = resample(scenario.feature_data,
                                                                                    scenario.performance_data,
                                                                                    n_samples=amount_of_training_instances,
                                                                                    random_state=fold)

        X, y = self.construct_dataset(resampled_scenario_feature_data, resampled_scenario_performances)

        return X, y

    def construct_dataset(self, instance_features, performances):
        performances = performances.iloc[:, :].to_numpy() if isinstance(performances, pd.DataFrame) else performances[:,
                                                                                                         :]

        # ignore all unsolvable training instances
        nan_mask = np.all(np.isnan(performances), axis=1)
        instance_features = instance_features[~nan_mask]
        performances = performances[~nan_mask]

        if isinstance(instance_features, pd.DataFrame):
            instance_features = instance_features.to_numpy()

        return instance_features, performances

    def get_name(self):
        name = 'new_baseline_' + self.distance_function
        return name
