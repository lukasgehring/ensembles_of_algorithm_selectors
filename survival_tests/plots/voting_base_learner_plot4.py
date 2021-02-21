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
    voting12346 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting2346 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_3_4_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting1346 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_3_4_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting1246 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_4_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting1236 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_6' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting1234 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection.approach, vbs_sbs.metric, voting_base_learner_selection.result, ((voting_base_learner_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection ON vbs_sbs.scenario_name = voting_base_learner_selection.scenario_name AND vbs_sbs.fold = voting_base_learner_selection.fold AND vbs_sbs.metric = voting_base_learner_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    fig = plt.figure(1, figsize=(8, 6))

    ax = fig.add_subplot(111)

    voting_normal = float(voting12346.result)

    plt.axhspan(voting_normal, 1, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.27  # the width of the bars
    ax.bar(1, voting2346.result, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, voting1346.result, width, color=color1, label='SUNNY')
    ax.bar(3, voting1246.result, width, color=color1, label='ISAC')
    ax.bar(4, voting1236.result, width, color=color1, label='SATzilla-11')
    ax.bar(5, voting1234.result, width, color=color1, label='SurvivalForestPAR10')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla'11", "SF-PAR10"])

    ax.text(1, float(voting2346.result), round(float(voting2346.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(voting1346.result), round(float(voting1346.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(voting1246.result), round(float(voting1246.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(voting1236.result), round(float(voting1236.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(5, float(voting1234.result), round(float(voting1234.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)

    # plt.xticks(rotation=45, ha='right')

    plt.title("Voting without base learners 'SF-Exp.' and 'Multiclass'")
    plt.xlabel("left out base learner")
    plt.ylabel("nPAR10")

    ax.set_ylim(bottom=0.36)
    ax.set_ylim(top=0.45)

    plt.show()

    fig.savefig("plotted/voting_base_learner4.pdf", bbox_inches='tight')


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