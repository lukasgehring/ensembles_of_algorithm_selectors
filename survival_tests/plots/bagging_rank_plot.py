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

    bagging = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY approach")
    bagging_ranking = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_ranking.approach, vbs_sbs.metric, bagging_ranking.result, ((bagging_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_ranking ON vbs_sbs.scenario_name = bagging_ranking.scenario_name AND vbs_sbs.fold = bagging_ranking.fold AND vbs_sbs.metric = bagging_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    bagging_averaging = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_ranking.approach, vbs_sbs.metric, bagging_ranking.result, ((bagging_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_ranking ON vbs_sbs.scenario_name = bagging_ranking.scenario_name AND vbs_sbs.fold = bagging_ranking.fold AND vbs_sbs.metric = bagging_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(7, 5))

    ax = fig.add_subplot(111)

    width = 0.2  # the width of the bars
    ax.bar(1, bagging.result, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, bagging_ranking.result[0], width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(3, bagging_averaging.result[0], width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(4, bagging_ranking.result[1], width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(5, bagging_averaging.result[1], width, color=color1, label='PerAlgorithmRegressor')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Majority Voting", "Ranked Voting", "Averaging", "Weighted Ranked Voting", "Weighted Averaging"])

    ax.set_ylim(bottom=0.38)
    ax.set_ylim(top=0.44)

    #plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    fig.savefig("plotted/bagging_ranking.pdf", bbox_inches='tight')


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