import operator
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

# open pre computed base learner
def open_file(scenario_name):
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

    final_data = final_data / sum(final_data)

    final_final_data = list()
    for i in range(feature_length):
        final_final_data.append((i, final_data[i]))

    return final_final_data

scenario_name = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU"]
num_algorithms = [11, 8, 4, 2, 20, 11, 8, 6, 29]
color1 = '#264653'
color2 = '#2a9d8f'
color3 = '#e76f51'
color4 = '#e9c46a'
color5 = '#251314'

for num_algorithms, name in zip(num_algorithms, scenario_name):
    data = open_file(name)
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
    fig = plt.figure(figsize=(15, 8))  # width:20, height:3
    name = name + ": Algorithms = " + str(num_algorithms)
    plt.title(name)
    for i in range(feature_length):
        if feature_length - indices[i] <= 7:
            plt.bar(i, importances[i], color=color3, align="center")
        else:
            plt.bar(i, importances[i], color=color1, align="center")

    plt.xlabel("Average feature ranking for all folds and algorithms of the scenario")
    plt.xlim([-1, len(final_data)])
    plt.show()

    filename = name + ".pdf"
    fig.savefig(filename, bbox_inches='tight')