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
    adaboostR2 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, AdaBoostR2.approach, vbs_sbs.metric, AdaBoostR2.result, ((AdaBoostR2.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN AdaBoostR2 ON vbs_sbs.scenario_name = AdaBoostR2.scenario_name AND vbs_sbs.fold = AdaBoostR2.fold AND vbs_sbs.metric = AdaBoostR2.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='adaboostR2_per_algorithm_regressor_10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    boosting = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting.approach, vbs_sbs.metric, boosting.result, ((boosting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting ON vbs_sbs.scenario_name = boosting.scenario_name AND vbs_sbs.fold = boosting.fold AND vbs_sbs.metric = boosting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='boosting' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    print(boosting)
    print(adaboostR2)
    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    width = 0.5  # the width of the bars
    ax.bar(1, adaboostR2.result, width, color=color1, label='AdaBoostR2')
    ax.bar(2, boosting.result, width, color=color1, label='Boosting')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["AdaBoostR2", "Boosting"])

    ax.text(1, float(adaboostR2.result), round(float(adaboostR2.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(boosting.result), round(float(boosting.result), 3), ha='center', va='bottom', rotation=0)

    plt.xticks(rotation=90)

    ax.set_ylim(bottom=0.36)
    #ax.set_ylim(top=0.44)

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