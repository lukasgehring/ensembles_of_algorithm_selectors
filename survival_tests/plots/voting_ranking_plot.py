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
    color3 = '#E4B363'
    color4 = '#9A8F97'
    color5 = '#EE4266'
    color6 = '#FB8B24'

    # voting 1234567
    voting1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_ranking_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")
    voting_ranking_max_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_max_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")
    voting_ranking_min_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_min_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")

    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_ranking_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")
    voting_ranking_max_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_max_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")
    voting_ranking_min_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_min_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(7, 5))

    ax = fig.add_subplot(111)

    width = 0.12  # the width of the bars
    ax.bar(0.79, voting1234567.result, width, color=color1, label='Majority Voting', zorder=3)
    ax.bar(0.93, voting_ranking_1234567.result, width, color=color2, label='Ranked Voting (avg)', zorder=3)
    ax.bar(1.07, voting_ranking_max_1234567.result, width, color=color5, label='Ranked Voting (max)', zorder=3)
    ax.bar(1.21, voting_ranking_min_1234567.result, width, color=color6, label='Ranked Voting (min)', zorder=3)

    ax.bar(1.59, voting24567.result, width, color=color1, zorder=3)
    ax.bar(1.73, voting_ranking_24567.result, width, color=color2, zorder=3)
    ax.bar(1.87, voting_ranking_max_24567.result, width, color=color5, zorder=3)
    ax.bar(2.01, voting_ranking_min_24567.result, width, color=color6, zorder=3)


    ax.set_xticks([1, 1.8])
    ax.set_xticklabels(["Voting", "Selected Voting"])

    ax.text(0.79, float(voting1234567.result), round(float(voting1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(0.93, float(voting_ranking_1234567.result), round(float(voting_ranking_1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.07, float(voting_ranking_max_1234567.result), round(float(voting_ranking_max_1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.21, float(voting_ranking_min_1234567.result), round(float(voting_ranking_min_1234567.result), 3), ha='center', va='bottom', rotation=0)

    ax.text(1.59, float(voting24567.result), round(float(voting24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.73, float(voting_ranking_24567.result), round(float(voting_ranking_24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.87, float(voting_ranking_max_24567.result), round(float(voting_ranking_max_24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2.01, float(voting_ranking_min_24567.result), round(float(voting_ranking_min_24567.result), 3), ha='center', va='bottom', rotation=0)

    ax.set_ylim(bottom=0.35)
    ax.set_ylim(top=0.44)

    ax.set_ylabel('nPAR10')

    plt.legend(loc=2)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)
    plt.show()

    fig.savefig("plotted/voting_ranking.pdf", bbox_inches='tight')


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