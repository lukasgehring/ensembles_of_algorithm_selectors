import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    bagging_with_ranking_color = '#2a9d8f'
    bagging_without_ranking_color_color = '#e9c46a'
    voting_weighting_cross_color = '#264653'

    bagging_with_ranking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_regression.approach, vbs_sbs.metric, bagging_regression.result, ((bagging_regression.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_regression ON vbs_sbs.scenario_name = bagging_regression.scenario_name AND vbs_sbs.fold = bagging_regression.fold AND vbs_sbs.metric = bagging_regression.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging10' OR approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_with_ranking' GROUP BY scenario_name")
    bagging_without_ranking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_without_ranking.approach, vbs_sbs.metric, bagging_without_ranking.result, ((bagging_without_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_without_ranking ON vbs_sbs.scenario_name = bagging_without_ranking.scenario_name AND vbs_sbs.fold = bagging_without_ranking.fold AND vbs_sbs.metric = bagging_without_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    plt.axhline(np.average(bagging_with_ranking.result), color=bagging_with_ranking_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(bagging_without_ranking.result), color=bagging_without_ranking_color_color, linestyle='dashed',
                linewidth=1)

    width = 0.2  # the width of the bars
    ind = np.arange(len(bagging_with_ranking.scenario_name))
    ax.bar(ind + 4 * width / 3, bagging_with_ranking.result, width, color=bagging_with_ranking_color,
           label='Bagging with ranking')
    ind = np.arange(len(bagging_without_ranking.scenario_name))
    ax.bar(ind - width / 2, bagging_without_ranking.result, width, color=bagging_without_ranking_color_color, label='Bagging without ranking')
    #ax.bar(ind + width / 2, voting.result, width, color=voting_color, label='Voting')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(bagging_without_ranking.scenario_name)

    #ax.plot(stacking.scenario_name, stacking.result, color=stacking_color, label='Stacking')
    #ax.plot(voting.scenario_name, voting.result, color=voting_color, label='Voting')
    #ax.axis([0, 24, 0, 50])
    plt.xticks(rotation=90)
    plt.legend()
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