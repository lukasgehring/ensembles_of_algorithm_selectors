import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    color1 = '#03071e'
    color2 = '#370617'
    color3 = '#6a040f'
    color4 = '#9d0208'
    color5 = '#d00000'
    color6 = '#dc2f02'
    color7 = '#e85d04'
    color8 = '#f48c06'
    color9 = '#faa307'
    color10 = '#ffba08'

    bagging_2 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_2_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_4 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_4_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_6 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_6_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_8 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_8_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_10 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test .approach, vbs_sbs.metric, final_bagging_base_learner_test .result, ((final_bagging_base_learner_test .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test  ON vbs_sbs.scenario_name = final_bagging_base_learner_test .scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test .fold AND vbs_sbs.metric = final_bagging_base_learner_test .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    bagging_12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_12_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_14 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_14_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_16 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_16_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_18 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_18_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_20 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_20_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_24 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_24_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_28 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_28_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_32 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_32_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_36 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_36_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_40 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_40_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    #plt.axhline(np.average(bagging_2.result), color=color1, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_4.result), color=color2, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_6.result), color=color3, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_8.result), color=color4, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_10.result), color=color5, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_12.result), color=color6, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_14.result), color=color7, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_16.result), color=color8, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_18.result), color=color9, linestyle='dashed', linewidth=2)
    #plt.axhline(np.average(bagging_20.result), color=color10, linestyle='dashed', linewidth=2)

    a = list()
    a.append(np.average(bagging_2.result))
    a.append(np.average(bagging_4.result))
    a.append(np.average(bagging_6.result))
    a.append(np.average(bagging_8.result))
    a.append(np.average(bagging_10.result))
    a.append(np.average(bagging_12.result))
    a.append(np.average(bagging_14.result))
    a.append(np.average(bagging_16.result))
    a.append(np.average(bagging_18.result))
    a.append(np.average(bagging_20.result))
    a.append(np.average(bagging_24.result))
    a.append(np.average(bagging_28.result))
    a.append(np.average(bagging_32.result))
    a.append(np.average(bagging_36.result))
    a.append(np.average(bagging_40.result))

    b = list()
    b.append("2")
    b.append("4")
    b.append("6")
    b.append("8")
    b.append("10")
    b.append("12")
    b.append("14")
    b.append("16")
    b.append("18")
    b.append("20")
    b.append("24")
    b.append("28")
    b.append("32")
    b.append("36")
    b.append("40")

    width = 0.5  # the width of the bars
    #ind = np.arange(len(bagging_2.scenario_name))
    ind = np.arange(15)

    #ax.bar(ind - 9 * width / 2, bagging_2.result, width, color=color1, label='2 Base Learner')
    #ax.bar(ind - 7 * width / 2, bagging_4.result, width, color=color2, label='4 Base Learner')
    #ax.bar(ind - 5 * width / 2, bagging_6.result, width, color=color3, label='6 Base Learner')
    #ax.bar(ind - 3 * width / 2, bagging_8.result, width, color=color4, label='8 Base Learner')
    #ax.bar(ind - width / 2, bagging_10.result, width, color=color5, label='10 Base Learner')
    #ax.bar(ind + width / 2, bagging_12.result, width, color=color6, label='20 Base Learner')
    #ax.bar(ind + 3 * width / 2, bagging_14.result, width, color=color7, label='30 Base Learner')
    #ax.bar(ind + 5 * width / 2, bagging_16.result, width, color=color8, label='40 Base Learner')
    #ax.bar(ind + 7 * width / 2, bagging_18.result, width, color=color9, label='50 Base Learner')
    #ax.bar(ind + 9 * width / 2, bagging_20.result, width, color=color10, label='60 Base Learner')

    ax.bar(ind + width / 2, a, width, color=color10, label='60 Base Learner')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(b)
    ax.set_ylim(bottom=0.35)

    #ax.plot(stacking.scenario_name, stacking.result, color=stacking_color, label='Stacking')
    #ax.plot(voting.scenario_name, voting.result, color=voting_color, label='Voting')
    #ax.axis([0, 24, 0, 50])
    plt.xticks(rotation=90)
    #plt.legend()
    plt.show()


def get_dataframe_for_sql_query(sql_query: str):
    db_credentials = get_database_credential_string()
    return pd.read_sql(sql_query, con=db_credentials)


def get_database_credential_string():
    config = load_configuration()
    db_config_section = config['DATABASE']
    db_host = db_config_section['host']
    db_username = db_config_section['username']
    db_password = db_config_section['password']
    db_database = db_config_section['database']
    return "mysql://" + db_username + ":" + db_password + "@" + db_host + "/" + db_database


config = load_configuration()
generate_sbs_vbs_change_table()