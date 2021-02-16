import operator
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def plot(algorithm):
    scenario_name = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU", "SAT18-EXP"]
    num_algorithms = [11, 8, 4, 2, 20, 11, 8, 6, 29, 5, 10, 31, 37]
    color1 = '#264653'
    color2 = '#f94144'
    color3 = '#f3722c'
    color4 = '#f9c74f'
    color5 = '#90be6d'
    color6 = '#43aa8b'
    color7 = '#4d908e'
    color8 = '#277da1'

    # Plot the impurity-based feature importances of the forest
    fig = plt.figure(1, figsize=(20, 10))
    plt_counter = 0

    for num_algorithms, name in zip(num_algorithms, scenario_name):
        plt_counter = plt_counter + 1
        ax = fig.add_subplot(3, 5, plt_counter)

        data = open_file(algorithm + name)
        print(name)
        final_data = avg_importance(data)
        final_data.sort(key=operator.itemgetter(1))

        feature_length = len(final_data)

        importances = np.zeros(feature_length)
        indices = np.zeros(feature_length)
        for index, t in enumerate(final_data):
            indices[index] = t[0]
            importances[index] = t[1]

        indices = indices.astype(int)

        name = name + ": Algorithms = " + str(num_algorithms)
        plt.title(name)

        tmp = np.arange(1, num_algorithms * 7 + 1)
        tmp = np.array_split(tmp, 7)
        g1 = tmp[0]
        g2 = tmp[1]
        g3 = tmp[2]
        g4 = tmp[3]
        g5 = tmp[4]
        g6 = tmp[5]
        g7 = tmp[6]


        for i in range(feature_length):
            if feature_length - indices[i] in g1:
                ax.bar(i, importances[i], color=color2, align="center")
            elif feature_length - indices[i] in g2:
                ax.bar(i, importances[i], color=color3, align="center")
            elif feature_length - indices[i] in g3:
                ax.bar(i, importances[i], color=color4, align="center")
            elif feature_length - indices[i] in g4:
                ax.bar(i, importances[i], color=color5, align="center")
            elif feature_length - indices[i] in g5:
                ax.bar(i, importances[i], color=color6, align="center")
            elif feature_length - indices[i] in g6:
                ax.bar(i, importances[i], color=color7, align="center")
            elif feature_length - indices[i] in g7:
                ax.bar(i, importances[i], color=color8, align="center")
            else:
                ax.bar(i, importances[i], color=color1, align="center")

        plt.xlabel("Average feature ranking for all folds and algorithms of the scenario")
        plt.xlim([-1, len(final_data)])

    l1 = mpatches.Patch(color=color1, label='Original Feature Data')
    l2 = mpatches.Patch(color=color2, label='Multiclass Predictions')
    l3 = mpatches.Patch(color=color3, label='SF-PAR10 Predictions')
    l4 = mpatches.Patch(color=color4, label='SF-Exp Predictions')
    l5 = mpatches.Patch(color=color5, label='SATzilla Predictions')
    l6 = mpatches.Patch(color=color6, label='ISAC Predictions')
    l7 = mpatches.Patch(color=color7, label='SUNNY Predictions')
    l8 = mpatches.Patch(color=color8, label='PerAlgo Predictions')
    fig.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8], loc=4, prop={'size': 15})

    plt.show()

    filename = algorithm + "feature_importance.pdf"
    fig.savefig(filename, bbox_inches='tight')

def single_plot(algorithm):
    #scenario_name = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU", "SAT18-EXP"]
    scenario_name = ["QBF-2011", "SAT12-INDU"]

    num_algorithms = [5, 31]
    color1 = '#264653'
    color2 = '#f94144'
    color3 = '#f3722c'
    color4 = '#f9c74f'
    color5 = '#90be6d'
    color6 = '#43aa8b'
    color7 = '#4d908e'
    color8 = '#277da1'

    for num_algorithms, name in zip(num_algorithms, scenario_name):
        # Plot the impurity-based feature importances of the forest
        fig = plt.figure(1, figsize=(10, 5))
        ax = fig.add_subplot(111)

        data = open_file(algorithm + name)
        print(name)
        final_data = avg_importance(data)
        final_data.sort(key=operator.itemgetter(1))

        feature_length = len(final_data)

        importances = np.zeros(feature_length)
        indices = np.zeros(feature_length)
        for index, t in enumerate(final_data):
            indices[index] = t[0]
            importances[index] = t[1]

        indices = indices.astype(int)

        name = name + ": Algorithms = " + str(num_algorithms)
        plt.title(name)

        tmp = np.arange(1, num_algorithms * 7 + 1)
        tmp = np.array_split(tmp, 7)
        g1 = tmp[0]
        g2 = tmp[1]
        g3 = tmp[2]
        g4 = tmp[3]
        g5 = tmp[4]
        g6 = tmp[5]
        g7 = tmp[6]


        for i in range(feature_length):
            if feature_length - indices[i] in g1:
                ax.bar(i, importances[i], color=color2, align="center")
            elif feature_length - indices[i] in g2:
                ax.bar(i, importances[i], color=color3, align="center")
            elif feature_length - indices[i] in g3:
                ax.bar(i, importances[i], color=color4, align="center")
            elif feature_length - indices[i] in g4:
                ax.bar(i, importances[i], color=color5, align="center")
            elif feature_length - indices[i] in g5:
                ax.bar(i, importances[i], color=color6, align="center")
            elif feature_length - indices[i] in g6:
                ax.bar(i, importances[i], color=color7, align="center")
            elif feature_length - indices[i] in g7:
                ax.bar(i, importances[i], color=color8, align="center")
            else:
                ax.bar(i, importances[i], color=color1, align="center")

        plt.xlabel("Average feature ranking for all folds and algorithms of the scenario")
        plt.xlabel("feature importance")
        plt.xlim([-1, len(final_data)])

        l1 = mpatches.Patch(color=color1, label='Original Feature Data')
        l2 = mpatches.Patch(color=color2, label='Multiclass Predictions')
        l3 = mpatches.Patch(color=color3, label='SF-PAR10 Predictions')
        l4 = mpatches.Patch(color=color4, label='SF-Exp Predictions')
        l5 = mpatches.Patch(color=color5, label='SATzilla Predictions')
        l6 = mpatches.Patch(color=color6, label='ISAC Predictions')
        l7 = mpatches.Patch(color=color7, label='SUNNY Predictions')
        l8 = mpatches.Patch(color=color8, label='PerAlgo Predictions')
        plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8], loc=2, prop={'size': 15})

        plt.show()

        filename = algorithm + "feature_importance%s.pdf" % name
        fig.savefig(filename, bbox_inches='tight')

#plot('multiclass')
plot('')
