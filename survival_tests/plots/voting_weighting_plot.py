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
    color3 = '#e76f51'
    color4 = '#e9c46a'
    color5 = '#251314'

    # Voting
    voting1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection .approach, vbs_sbs.metric, voting_base_learner_selection .result, ((voting_base_learner_selection .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection  ON vbs_sbs.scenario_name = voting_base_learner_selection .scenario_name AND vbs_sbs.fold = voting_base_learner_selection .fold AND vbs_sbs.metric = voting_base_learner_selection .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_weighting_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_cross_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_cross.approach, vbs_sbs.metric, voting_cross.result, ((voting_cross.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_cross ON vbs_sbs.scenario_name = voting_cross.scenario_name AND vbs_sbs.fold = voting_cross.fold AND vbs_sbs.metric = voting_cross.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_cross_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    # Selected Voting
    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection .approach, vbs_sbs.metric, voting_base_learner_selection .result, ((voting_base_learner_selection .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection  ON vbs_sbs.scenario_name = voting_base_learner_selection .scenario_name AND vbs_sbs.fold = voting_base_learner_selection .fold AND vbs_sbs.metric = voting_base_learner_selection .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_weighting_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_cross_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_cross.approach, vbs_sbs.metric, voting_cross.result, ((voting_cross.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_cross ON vbs_sbs.scenario_name = voting_cross.scenario_name AND vbs_sbs.fold = voting_cross.fold AND vbs_sbs.metric = voting_cross.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_cross_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(7, 5))

    ax = fig.add_subplot(111)

    width = 0.25  # the width of the bars
    ax.bar(0.7, voting1234567.result, width, color=color1, zorder=6)
    ax.bar(1, voting_weighting_1234567.result, width, color=color2, zorder=6)
    ax.bar(1.3, voting_cross_1234567.result, width, color=color3, zorder=6)

    ax.bar(1.7, voting24567.result, width, color=color1, zorder=6)
    ax.bar(2, voting_weighting_24567.result, width, color=color2, zorder=6)
    ax.bar(2.3, voting_cross_24567.result, width, color=color3, zorder=6)

    ax.text(0.7, float(voting1234567.result), round(float(voting1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1, float(voting_weighting_1234567.result), round(float(voting_weighting_1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.3, float(voting_cross_1234567.result), round(float(voting_cross_1234567.result), 3), ha='center', va='bottom', rotation=0)

    ax.text(1.7, float(voting24567.result), round(float(voting24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(voting_weighting_24567.result), round(float(voting_weighting_24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2.3, float(voting_cross_24567.result), round(float(voting_cross_24567.result), 3), ha='center', va='bottom', rotation=0)


    ax.set_xticks([1,2])
    ax.set_xticklabels(
        ["Voting", "Selected Voting"])

    #plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('nPAR10', fontsize=11)

    ax.set_ylim(bottom=0.35)
    ax.set_ylim(top=0.42)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    l1 = mpatches.Patch(color=color1, label="Majority Voting")
    l2 = mpatches.Patch(color=color2, label="Weighted Voting")
    l3 = mpatches.Patch(color=color3, label="Weighted Voting with cross-validation")

    fig.legend(handles=[l1, l2, l3], loc=1, prop={'size': 13}, bbox_to_anchor=(0.99, 0.98))

    plt.legend()
    plt.show()

    fig.savefig("plotted/voting_cross.pdf", bbox_inches='tight')


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