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
    bagging = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")
    bagging_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_weighting.approach, vbs_sbs.metric, bagging_weighting.result, ((bagging_weighting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_weighting ON vbs_sbs.scenario_name = bagging_weighting.scenario_name AND vbs_sbs.fold = bagging_weighting.fold AND vbs_sbs.metric = bagging_weighting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking_weighting' GROUP BY scenario_name")
    bagging_rank = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_ranking.approach, vbs_sbs.metric, bagging_ranking.result, ((bagging_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_ranking ON vbs_sbs.scenario_name = bagging_ranking.scenario_name AND vbs_sbs.fold = bagging_ranking.fold AND vbs_sbs.metric = bagging_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_with_ranking' GROUP BY scenario_name")
    bagging_average_rank = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_averaging.approach, vbs_sbs.metric, bagging_averaging.result, ((bagging_averaging.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_averaging ON vbs_sbs.scenario_name = bagging_averaging.scenario_name AND vbs_sbs.fold = bagging_averaging.fold AND vbs_sbs.metric = bagging_averaging.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_with_ranking_averaging' GROUP BY scenario_name")

    bagging_weight_rank = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_ranking.approach, vbs_sbs.metric, bagging_ranking.result, ((bagging_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_ranking ON vbs_sbs.scenario_name = bagging_ranking.scenario_name AND vbs_sbs.fold = bagging_ranking.fold AND vbs_sbs.metric = bagging_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_with_ranking_weighting' GROUP BY scenario_name")
    bagging_average_weighting_rank = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_averaging.approach, vbs_sbs.metric, bagging_averaging.result, ((bagging_averaging.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_averaging ON vbs_sbs.scenario_name = bagging_averaging.scenario_name AND vbs_sbs.fold = bagging_averaging.fold AND vbs_sbs.metric = bagging_averaging.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_with_ranking_averaging_weighting' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    # PerAlgorithmRegressor
    frames = [bagging.result, bagging_weighting.result, bagging_rank.result, bagging_average_rank.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(2).T
    table = result.to_latex()
    table = table.replace("\n0  ", "\nASP-POTASSCO          ")
    table = table.replace("\n1  ", "\nBNSL-2016             ")
    table = table.replace("\n2  ", "\nCPMP-2015             ")
    table = table.replace("\n3  ", "\nCSP-2010              ")
    table = table.replace("\n4  ", "\nCSP-Minizinc-Time-2016")
    table = table.replace("\n5  ", "\nCSP-MZN-2013          ")
    table = table.replace("\n6  ", "\nGLUHACK-18            ")
    table = table.replace("\n7  ", "\nMAXSAT12-PMS          ")
    table = table.replace("\n8  ", "\nMAXSAT15-PMS-INDU     ")
    table = table.replace("\n9  ", "\nQBF-2011              ")
    table = table.replace("\n10 ", "\nSAT03-16\_INDU        ")
    table = table.replace("\n11 ", "\nSAT12-INDU            ")
    table = table.replace("\n12 ", "\nSAT18-EXP             ")
    print(table)

    # 2
    frames = [bagging.result, bagging_weight_rank.result, bagging_average_weighting_rank.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(2).T
    table = result.to_latex()
    table = table.replace("\n0  ", "\nASP-POTASSCO          ")
    table = table.replace("\n1  ", "\nBNSL-2016             ")
    table = table.replace("\n2  ", "\nCPMP-2015             ")
    table = table.replace("\n3  ", "\nCSP-2010              ")
    table = table.replace("\n4  ", "\nCSP-Minizinc-Time-2016")
    table = table.replace("\n5  ", "\nCSP-MZN-2013          ")
    table = table.replace("\n6  ", "\nGLUHACK-18            ")
    table = table.replace("\n7  ", "\nMAXSAT12-PMS          ")
    table = table.replace("\n8  ", "\nMAXSAT15-PMS-INDU     ")
    table = table.replace("\n9  ", "\nQBF-2011              ")
    table = table.replace("\n10 ", "\nSAT03-16\_INDU        ")
    table = table.replace("\n11 ", "\nSAT12-INDU            ")
    table = table.replace("\n12 ", "\nSAT18-EXP             ")
    print(table)


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