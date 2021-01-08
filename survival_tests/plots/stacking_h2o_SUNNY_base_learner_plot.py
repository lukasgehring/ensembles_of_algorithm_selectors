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
    stacking1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_2_3_4_5_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__2_3_4_5_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking134567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_3_4_5_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking124567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_2_4_5_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking123567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_2_3_5_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking123467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_2_3_4_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking123457 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_2_3_4_5_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking123456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_h2o.approach, vbs_sbs.metric, stacking_h2o.result, ((stacking_h2o.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_h2o ON vbs_sbs.scenario_name = stacking_h2o.scenario_name AND vbs_sbs.fold = stacking_h2o.fold AND vbs_sbs.metric = stacking_h2o.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking__1_2_3_4_5_6SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    fig, ax = plt.subplots()

    voting_normal = float(stacking1234567.result)

    plt.axhspan(voting_normal, 1, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking234567.result, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, stacking134567.result, width, color=color1, label='SUNNY')
    ax.bar(3, stacking124567.result, width, color=color1, label='ISAC')
    ax.bar(4, stacking123567.result, width, color=color1, label='SATzilla-11')
    ax.bar(5, stacking123467.result, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(6, stacking123457.result, width, color=color1, label='SurvivalForestPAR10')
    ax.bar(7, stacking123456.result, width, color=color1, label='Multiclass')

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])

    ax.text(1, float(stacking234567.result), round(float(stacking234567.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(stacking134567.result), round(float(stacking134567.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(stacking124567.result), round(float(stacking124567.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(stacking123567.result), round(float(stacking123567.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(5, float(stacking123467.result), round(float(stacking123467.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(6, float(stacking123457.result), round(float(stacking123457.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(7, float(stacking123456.result), round(float(stacking123456.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)

    ax.text(7.9, voting_normal - 0.002, round(voting_normal, 3), ha='center', va='bottom', rotation=0)
    #plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=0.42)
    ax.set_ylim(top=0.48)

    plt.title("Stacking with base learner 1,2,3,4,5,6,7")
    plt.xlabel("left out base learner")
    plt.ylabel("nPAR10")

    plt.show()

    fig.savefig("1.pdf", bbox_inches='tight')


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