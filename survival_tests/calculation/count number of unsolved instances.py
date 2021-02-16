from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np

scenario = ASlibScenario()
scenario.read_scenario('../data/aslib_data-master/SAT18-EXP')
unsolved = 0
for x in scenario.performance_data.to_numpy():
    min = np.amin(x)
    if min > scenario.algorithm_cutoff_time:
        unsolved = unsolved + 1
print(unsolved)