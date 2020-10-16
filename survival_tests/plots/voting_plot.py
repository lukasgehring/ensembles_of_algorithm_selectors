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
    voting_weighting_color = '#e9c46a'
    voting_weighting_cross_color = '#264653'

    voting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting.approach, vbs_sbs.metric, voting.result, ((voting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting ON vbs_sbs.scenario_name = voting.scenario_name AND vbs_sbs.fold = voting.fold AND vbs_sbs.metric = voting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting.approach, vbs_sbs.metric, voting_weighting.result, ((voting_weighting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting ON vbs_sbs.scenario_name = voting_weighting.scenario_name AND vbs_sbs.fold = voting_weighting.fold AND vbs_sbs.metric = voting_weighting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting_weighting_cross = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_cross.approach, vbs_sbs.metric, voting_weighting_cross.result, ((voting_weighting_cross.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_cross ON vbs_sbs.scenario_name = voting_weighting_cross.scenario_name AND vbs_sbs.fold = voting_weighting_cross.fold AND vbs_sbs.metric = voting_weighting_cross.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    print(voting_weighting)

    print("")

    print(voting_weighting_cross)

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    plt.axhline(np.average(voting_weighting.result), color=voting_weighting_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(voting.result), color=voting_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(voting_weighting_cross.result), color=voting_weighting_cross_color, linestyle='dashed', linewidth=1)

    width = 0.2  # the width of the bars
    ind = np.arange(len(voting_weighting_cross.scenario_name))
    ax.bar(ind + 4 * width / 3, voting_weighting_cross.result, width, color=voting_weighting_cross_color,
           label='Voting Weighting with Cross Validation')

    ind = np.arange(len(voting_weighting.scenario_name))
    ax.bar(ind - width / 2, voting_weighting.result, width, color=voting_weighting_color, label='Voting Weighting')
    ax.bar(ind + width / 2, voting.result, width, color=voting_color, label='Voting')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(voting_weighting.scenario_name)

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