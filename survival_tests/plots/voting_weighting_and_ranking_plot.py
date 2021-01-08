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
    color5 = '#251314'

    #TODO: correct version for voting normal??
    voting1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_weighting_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")
    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_weighting_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")
    voting12346 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_weighting_12346 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_1_2_3_4_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach  ")


    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    width = 0.18  # the width of the bars
    ax.bar(0.8, voting1234567.result, width, color=color1, label='Majority Voting')

    ax.bar(1, voting_weighting_1234567.result, width, color=color2, label='Ranked Voting (avg ranking)')

    ax.bar(1.8, voting24567.result, width, color=color3, label='Weighted Voting')

    ax.bar(2, voting_weighting_24567.result, width, color=color4, label='Weighted-Ranked Voting (avg ranking)')

    ax.bar(2.8, voting12346.result, width, color=color1)

    ax.bar(3, voting_weighting_12346.result, width, color=color2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["f_voting", "s1_voting", "s2_voting"])

    ax.text(0.8, float(voting1234567.result), round(float(voting1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1, float(voting_weighting_1234567.result), round(float(voting_weighting_1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.8, float(voting24567.result), round(float(voting24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(voting_weighting_24567.result), round(float(voting_weighting_24567.result), 3), ha='center', va='bottom', rotation=0)

    ax.text(2.8, float(voting12346.result), round(float(voting12346.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(voting_weighting_12346.result), round(float(voting_weighting_12346.result), 3), ha='center', va='bottom', rotation=0)

    #plt.xticks(rotation=90)

    ax.set_ylim(bottom=0.34)
    ax.set_ylim(top=0.42)

    ax.set_ylabel('nPAR10')

    plt.legend()

    plt.show()

    fig.savefig("voting_weighting.pdf", bbox_inches='tight')


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