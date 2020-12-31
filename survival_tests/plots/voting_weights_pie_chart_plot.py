import numpy as np
import matplotlib.pyplot as plt


def plot_scenario(scenario):
    with open('../weights/%s.csv' % scenario) as csvfile:
        folds = csvfile.read().replace(']', '').split('[')
        num_base_learner = 5
        avg_weights = np.zeros(num_base_learner)
        for fold in folds:
            for i, weights in enumerate(fold.split(',')):
                if i > 1:
                    avg_weights[i - 2] = avg_weights[i - 2] + float(weights) / 10

        avg_weights = avg_weights / sum((avg_weights))


    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = ['SUNNY', 'SATzilla11', 'Expectation', 'PAR10', 'Multiclass']
    sizes = avg_weights
    explode = [0, 0, 0, 0, 0]  # only "explode" the 2nd slice (i.e. 'Hogs')
    explode[np.argmax(avg_weights)] = 0.1

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title(scenario)

    plt.show()

scenarios = ['ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT15-PMS-INDU']

for s in scenarios:
    plot_scenario(s)


