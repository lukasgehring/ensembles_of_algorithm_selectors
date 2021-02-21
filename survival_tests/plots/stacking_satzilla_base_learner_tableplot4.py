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
    stackingcomp = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_4_5_6SATzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking1 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_4_5_6SATzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking4 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_5_6SATzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking5 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_4_6SATzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking6 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_4_5SATzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    fig, ax = plt.subplots()

    voting_normal = float(stackingcomp.result)

    plt.axhspan(voting_normal, 1, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking1.result, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, stacking4.result, width, color=color1, label='SATzilla-11')
    ax.bar(3, stacking5.result, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(4, stacking6.result, width, color=color1, label='SurvivalForestPAR10')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["PerAlgo", "SATzilla'11", "SF-Exp.", "SF-PAR10"])

    ax.text(1, float(stacking1.result), round(float(stacking1.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(stacking4.result), round(float(stacking4.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(stacking5.result), round(float(stacking5.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(stacking6.result), round(float(stacking6.result) - voting_normal, 3), ha='center', va='bottom', rotation=0)

    ax.text(4.6, voting_normal - 0.002, round(voting_normal, 3), ha='center', va='bottom', rotation=0)
    #plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=0.38)
    ax.set_ylim(top=0.5)

    plt.title("Stacking without base learners 'ISAC', 'Multiclass' and 'SUNNY'")
    plt.xlabel("left out base learner")
    plt.ylabel("nPAR10")

    plt.show()

    fig.savefig("plotted/stacking_base_learner4_sat.pdf", bbox_inches='tight')


    # table
    frames = [stackingcomp.result, stacking1.result, stacking4.result, stacking5.result, stacking6.result]
    result = pd.concat(frames, axis=1).T
    #result['average'] = result.mean(numeric_only=True, axis=1)
    #result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(4)
    table = result.to_latex()
    print(table)


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