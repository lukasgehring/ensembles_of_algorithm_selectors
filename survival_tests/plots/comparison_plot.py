import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    voting_color = '#2a9d8f'
    stacking_color = '#e9c46a'
    run2survive_color = '#e76f51'
    bagging_color = '#264653'
    
    stacking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking .approach, vbs_sbs.metric, stacking .result, ((stacking .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking  ON vbs_sbs.scenario_name = stacking .scenario_name AND vbs_sbs.fold = stacking .fold AND vbs_sbs.metric = stacking .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_2_4_5_6SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting_weighting_ranking_24567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting.approach, vbs_sbs.metric, voting_weighting.result, ((voting_weighting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting ON vbs_sbs.scenario_name = voting_weighting.scenario_name AND vbs_sbs.fold = voting_weighting.fold AND vbs_sbs.metric = voting_weighting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_weighting_2_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    boosting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting.approach, vbs_sbs.metric, boosting.result, ((boosting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting ON vbs_sbs.scenario_name = boosting.scenario_name AND vbs_sbs.fold = boosting.fold AND vbs_sbs.metric = boosting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    bagging_40 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_40_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    ax.text(13.8, np.average(stacking.result) - 0.02, round(np.average(stacking.result), 2), ha='center', va='bottom', rotation=0)
    plt.axhline(np.average(stacking.result), color=stacking_color, linestyle='dashed', linewidth=1)

    ax.text(13.8, np.average(boosting.result) - 0.035, round(np.average(boosting.result), 2), ha='center', va='bottom', rotation=0)
    plt.axhline(np.average(boosting.result), color=run2survive_color, linestyle='dashed', linewidth=1)

    ax.text(13.8, np.average(bagging_40.result) - 0.02, round(np.average(bagging_40.result), 2), ha='center', va='bottom', rotation=0)
    plt.axhline(np.average(bagging_40.result), color=bagging_color, linestyle='dashed', linewidth=1)

    ax.text(13.8, np.average(voting_weighting_ranking_24567.result) - 0.04, round(np.average(voting_weighting_ranking_24567.result), 2), ha='center', va='bottom', rotation=0)
    plt.axhline(np.average(voting_weighting_ranking_24567.result), color=voting_color, linestyle='dashed', linewidth=1)

    ind = np.arange(len(stacking.scenario_name))
    width = 0.2  # the width of the bars
    ax.bar(ind, stacking.result, width, color=stacking_color, label='Stacking')
    ax.bar(ind - width, voting_weighting_ranking_24567.result, width, color=voting_color, label='Voting with cross validation')
    ax.bar(ind + width, boosting.result, width, color=run2survive_color, label='Run2Survive')
    ax.bar(ind + 2 * width, bagging_40.result, width, color=bagging_color, label='Bagging')


    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(stacking.scenario_name)

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