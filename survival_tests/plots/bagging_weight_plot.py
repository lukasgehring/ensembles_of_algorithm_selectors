import pandas as pd
import configparser
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    color1 = '#264653'
    color2 = '#2a9d8f'
    color3 = '#E4B363'
    color4 = '#9A8F97'
    color5 = '#251314'

    bagging = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY approach")
    bagging_weighting = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_weighting.approach, vbs_sbs.metric, bagging_weighting.result, ((bagging_weighting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_weighting ON vbs_sbs.scenario_name = bagging_weighting.scenario_name AND vbs_sbs.fold = bagging_weighting.fold AND vbs_sbs.metric = bagging_weighting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(7, 5))

    ax = fig.add_subplot(111)

    number_of_instances = 16484

    width = 0.18  # the width of the bars
    ax.bar(0.7, bagging.result, width, color=color1, zorder=6)
    ax.bar(0.9, bagging_weighting.result[0], width, color=color2, zorder=6)
    ax.bar(1.1, bagging_weighting.result[1], width, color=color3, zorder=6)
    ax.bar(1.3, bagging_weighting.result[2], width, color=color4, zorder=6)
    ax.bar(1.7, bagging_weighting.result[3], width, color=color1, zorder=6)
    ax.bar(1.9, bagging_weighting.result[4], width, color=color2, zorder=6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["PerAlgo", "SUNNY"])


    #plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('nPAR10', fontsize=11)
    ax.set_ylabel('Lernalgorithm', fontsize=11)

    ax.set_ylim(bottom=0.36)
    ax.set_ylim(top=0.46)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    l1 = mpatches.Patch(color=color1, label="Majority Voting")
    l2 = mpatches.Patch(color=color2, label="Weighted Voting")
    l3 = mpatches.Patch(color=color3, label="Weighted Voting (out-of-sample)")
    l4 = mpatches.Patch(color=color4, label="Ranked Voting (original-data)")

    plt.legend(handles=[l1, l2, l3, l4], loc=2)
    plt.show()

    fig.savefig("plotted/bagging_weighting.pdf", bbox_inches='tight')


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