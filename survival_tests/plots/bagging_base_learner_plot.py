import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    color1 = '#264653'
    color2 = '#2a9d8f'
    color3 = '#e76f51'
    color4 = '#e9c46a'
    color5 = '#251314'

    # PerAlgorithmRegressor
    bagging_4 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_4_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_8 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_8_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_12_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_16 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_16_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
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
    bagging_44 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_44_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_48 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_48_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_52 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_52_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_56 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_56_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_60 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_60_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")

    # SUNNY
    SUNNY_4 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_4_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_8 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_8_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_12_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_16 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_16_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_20 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_20_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_24 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_24_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_28 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_28_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_32 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_32_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_36 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_36_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_40 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_40_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_44 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_44_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_48 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_48_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_52 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_52_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_56 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_56_SUNNY_without_ranking' GROUP BY scenario_name")
    SUNNY_60 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_60_SUNNY_without_ranking' GROUP BY scenario_name")


    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    # PerAlgorithmRegressor
    frames = [bagging_4.result, bagging_8.result, bagging_12.result, bagging_16.result, bagging_20.result,
              bagging_24.result, bagging_28.result, bagging_32.result, bagging_36.result, bagging_40.result,
              bagging_44.result, bagging_48.result, bagging_52.result, bagging_56.result, bagging_60.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(6).T
    table = result.to_latex()
    table.replace("0 ", "ASP-POTASSCO ")
    table.replace("1 ", "BNSL-2016 ")
    table.replace("2 ", "CPMP-2015 ")
    table.replace("3 ", "CSP-2010 ")
    table.replace("4 ", "CSP-Minizinc-Time-2016 ")
    table.replace("5 ", "CSP-MZN-2013 ")
    table.replace("6 ", "GLUHACK-18 ")
    table.replace("7 ", "MAXSAT12-PMS ")
    table.replace("8 ", "MAXSAT15-PMS-INDU ")
    table.replace("9 ", "QBF-2011 ")
    table.replace("10 ", "SAT03-16\_INDU ")
    table.replace("11 ", "SAT12-INDU ")
    table.replace("12 ", "SAT18-EXP ")
    print(table)

    # SUNNY
    frames = [SUNNY_4.result, SUNNY_8.result, SUNNY_12.result, SUNNY_16.result, SUNNY_20.result,
              SUNNY_24.result, SUNNY_28.result, SUNNY_32.result, SUNNY_36.result, SUNNY_40.result,
              SUNNY_44.result, SUNNY_48.result, SUNNY_52.result, SUNNY_56.result, SUNNY_60.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(6).T
    table = result.to_latex()
    table.replace("0 ", "ASP-POTASSCO ")
    table.replace("1 ", "BNSL-2016 ")
    table.replace("2 ", "CPMP-2015 ")
    table.replace("3 ", "CSP-2010 ")
    table.replace("4 ", "CSP-Minizinc-Time-2016 ")
    table.replace("5 ", "CSP-MZN-2013 ")
    table.replace("6 ", "GLUHACK-18 ")
    table.replace("7 ", "MAXSAT12-PMS ")
    table.replace("8 ", "MAXSAT15-PMS-INDU ")
    table.replace("9 ", "QBF-2011 ")
    table.replace("10 ", "SAT03-16\_INDU ")
    table.replace("11 ", "SAT12-INDU ")
    table.replace("12 ", "SAT18-EXP ")
    print(table)


    a = list()
    a.append(np.average(bagging_4.result))
    a.append(np.average(bagging_8.result))
    a.append(np.average(bagging_12.result))
    a.append(np.average(bagging_16.result))
    a.append(np.average(bagging_20.result))
    a.append(np.average(bagging_24.result))
    a.append(np.average(bagging_28.result))
    a.append(np.average(bagging_32.result))
    a.append(np.average(bagging_36.result))
    a.append(np.average(bagging_40.result))
    a.append(np.average(bagging_44.result))
    a.append(np.average(bagging_48.result))
    a.append(np.average(bagging_52.result))
    a.append(np.average(bagging_56.result))
    a.append(np.average(bagging_60.result))

    c = list()
    c.append(np.average(SUNNY_4.result))
    c.append(np.average(SUNNY_8.result))
    c.append(np.average(SUNNY_12.result))
    c.append(np.average(SUNNY_16.result))
    c.append(np.average(SUNNY_20.result))
    c.append(np.average(SUNNY_24.result))
    c.append(np.average(SUNNY_28.result))
    c.append(np.average(SUNNY_32.result))
    c.append(np.average(SUNNY_36.result))
    c.append(np.average(SUNNY_40.result))
    c.append(np.average(SUNNY_44.result))
    c.append(np.average(SUNNY_48.result))
    c.append(np.average(SUNNY_52.result))
    c.append(np.average(SUNNY_56.result))
    c.append(np.average(SUNNY_60.result))

    b = list()
    b.append("4")
    b.append("8")
    b.append("12")
    b.append("16")
    b.append("20")
    b.append("24")
    b.append("28")
    b.append("32")
    b.append("36")
    b.append("40")
    b.append("44")
    b.append("48")
    b.append("52")
    b.append("56")
    b.append("60")

    width = 0.5  # the width of the bars
    ind = np.arange(len(b))


    ax.bar(ind + width / 2, a, width, color=color1, label='60 Base Learner')
    ax.bar(ind - width / 2, c, width, color=color2, label='60 Base Learner')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(b)
    #ax.set_ylim(bottom=0.36)

    plt.xlabel('Number of Base Learner')
    plt.ylabel('nPAR10')

    # 100 linearly spaced numbers
    x = np.linspace(0, 14, 15)
    print(x)

    # approximation plot for PerAlgorithmRegressor
    y = 0.0007905*(x** 2) + (-0.01650208)*x + 0.47456491
    ax.plot(x, y, 'r', color= color3)

    # approximation plot for SUNNY
    y = 0.0009834 * (x ** 2) + (-0.018609) * x + 0.46738839
    ax.plot(x, y, 'r', color= color4)

    ax.set_ylim(bottom=0.35)
    ax.set_ylim(top=0.5)

    plt.xticks(rotation=90)
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