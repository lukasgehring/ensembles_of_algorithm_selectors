import numpy as np
from scipy.stats import rankdata

def predict_with_ranking(features_of_test_instance, instance_id: int, num_algorithms: int, trained_models, weights=None, log=False):
    # use all predictions (ranking) of the base learner
    predictions = np.zeros((num_algorithms, 1))
    for i, model in enumerate(trained_models):
        # rank output from the base learner (best algorithm gets rank 1 - worst algorithm gets rank len(algorithms))
        if weights:
            if log:
                predictions = predictions.reshape(num_algorithms) + weights[i] * np.log(
                    rankdata(model.predict(features_of_test_instance, instance_id))) * (-1)
            else:
                predictions = predictions.reshape(num_algorithms) + weights[i] * rankdata(
                    model.predict(features_of_test_instance, instance_id)).reshape(num_algorithms)
        else:
            if log:
                predictions = predictions.reshape(num_algorithms) + np.log(
                    rankdata(model.predict(features_of_test_instance, instance_id))) * (-1)
            else:
                predictions = predictions.reshape(num_algorithms) + rankdata(
                    model.predict(features_of_test_instance, instance_id))

    return predictions / sum(predictions)