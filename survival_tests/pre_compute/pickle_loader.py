import pickle


# save base learner for later use
def save_prediction(self, base_learner_name, features_of_test_instance, prediction):
    file_name = 'predictions/' + base_learner_name + '_' + self.scenario_name + '_' + str(self.fold)
    with open(file_name, 'wb') as output:
        self.pred[str(features_of_test_instance)] = prediction
        pickle.dump(self.pred, output)


# save base learner for later use
def read_prediction(self):
    file_name = 'predictions/' + self.algorithm + '_' + self.scenario_name + '_' + str(self.fold)
    with open(file_name, 'rb') as input:
        return pickle.load(input)
