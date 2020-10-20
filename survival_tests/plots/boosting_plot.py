import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    boosting_color = '#2a9d8f'
    adaboost_color = '#e9c46a'

    boosting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting.approach, vbs_sbs.metric, boosting.result, ((boosting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting ON vbs_sbs.scenario_name = boosting.scenario_name AND vbs_sbs.fold = boosting.fold AND vbs_sbs.metric = boosting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='boosting' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    adaboost = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboost.approach, vbs_sbs.metric, adaboost.result, ((adaboost.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboost ON vbs_sbs.scenario_name = adaboost.scenario_name AND vbs_sbs.fold = adaboost.fold AND vbs_sbs.metric = adaboost.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    plt.axhline(np.average(adaboost.result), color=adaboost_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(boosting.result), color=boosting_color, linestyle='dashed', linewidth=1)

    width = 0.2  # the width of the bars
    ind = np.arange(len(adaboost.scenario_name))
    ax.bar(ind + 4 * width / 3, adaboost.result, width, color=adaboost_color,
           label='Adaboost')

    ind = np.arange(len(boosting.scenario_name))
    ax.bar(ind + width / 2, boosting.result, width, color=boosting_color, label='Boosting')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(boosting.scenario_name)

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