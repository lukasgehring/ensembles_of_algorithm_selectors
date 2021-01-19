import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scenario():
    scenarios = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013",
                 "GLUHACK-18", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU",
                 "SAT18-EXP"]

    values = []
    for scenario in scenarios:
        with open('../weights/%s.csv' % scenario) as csvfile:
            folds = csvfile.read().split('\n')[:-1]
            num_base_learner = 7
            avg_weights = np.zeros(num_base_learner)
            for fold in folds:
                fold = fold[1:-1].split(', ')
                s = fold[1]
                approach = fold[2]
                fold = fold[2:]
                print(fold)
                for i, weights in enumerate(fold):
                    avg_weights[i] = avg_weights[i] + float(weights) / 10

            avg_weights = avg_weights / sum((avg_weights)) * 100


        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        values.append((avg_weights))

    labels = ['PerAlgo', 'SUNNY', 'ISAC', 'SATzilla', 'SF-Exp', 'SF-PAR10', 'Multiclass']
    df = pd.DataFrame(values, columns=labels, index=scenarios)
    df.loc['mean'] = df.mean()
    df.loc['median'] = df.median()
    print(df.round(2).to_latex())

plot_scenario()

