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

    #TODO: correct version for voting normal??
    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    voting4567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting2567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting2467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting2457 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting2456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    voting_normal = float(voting24567.result)

    plt.axhspan(voting_normal, 1, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, voting4567.result, width, color=color1, label='SUNNY')
    ax.bar(2, voting2567.result, width, color=color1, label='SATzilla-11')
    ax.bar(3, voting2467.result, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(4, voting2457.result, width, color=color1, label='SurvivalForestPAR10')
    ax.bar(5, voting2456.result, width, color=color1, label='Multiclass')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["2", "4", "5", "6", "7"])

    ax.text(1, float(voting4567.result), round(float(voting4567.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(voting2567.result), round(float(voting2567.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(voting2467.result), round(float(voting2467.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(voting2457.result), round(float(voting2457.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(5, float(voting2456.result), round(float(voting2456.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)

    ax.text(5.7, voting_normal - 0.002, round(voting_normal, 3), ha='center', va='bottom', rotation=0)
    # plt.xticks(rotation=45, ha='right')

    plt.title("Voting with base learner 2,4,5,6,7")
    plt.xlabel("left out base learner")
    plt.ylabel("nPAR10")

    ax.set_ylim(bottom=0.36)
    ax.set_ylim(top=0.46)

    plt.show()

    fig.savefig("3.pdf", bbox_inches='tight')


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