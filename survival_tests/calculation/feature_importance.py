import operator
import pickle
import numpy as np
import matplotlib.pyplot as plt

# open pre computed base learner
def open_base_learner(scenario_name):
    file_name = '../feature_importance/' + scenario_name

    # code from https://stackoverflow.com/questions/35067957/how-to-read-pickle-file by jsbueno
    data = []
    with (open(file_name, "rb")) as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    return data

def avg_importance(data):
    feature_length = data[0][1]
    final_data = np.zeros(feature_length)
    for data_index, data_value in data:
        if data_index > -1:
            final_data[data_index] = final_data[data_index] + data_value
    final_data = final_data / feature_length

    final_final_data = list()
    for i in range(feature_length):
        final_final_data.append((i, final_data[i]))

    return final_final_data

scenario_name = ["QBF-2011"]
for name in scenario_name:
    data = open_base_learner(name)
    final_data = avg_importance(data)
    final_data.sort(key=operator.itemgetter(1))

    feature_length = len(final_data)

    importances = np.zeros(feature_length)
    indices = np.zeros(feature_length)
    for index, t in enumerate(final_data):
        indices[index] = t[0]
        importances[index] = t[1]

    indices = indices.astype(int)

    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize=(15, 8))  # width:20, height:3
    plt.title(name)
    for i in range(feature_length):
        if feature_length - indices[i] < 6:
            plt.bar(i, importances[i], color="b", align="center")
        else:
            plt.bar(i, importances[i], color="r", align="center")
    plt.xticks(range(len(final_data)), indices)
    plt.xlim([-1, len(final_data)])
    plt.show()