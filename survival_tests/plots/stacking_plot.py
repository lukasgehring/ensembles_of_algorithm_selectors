import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    color1 = '#2a9d8f'
    color2 = '#e9c46a'
    color3 = '#e76f51'
    color4 = '#264653'
    color5 = '#251314'

    stacking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_new.approach, vbs_sbs.metric, stacking_new.result, ((stacking_new.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_new ON vbs_sbs.scenario_name = stacking_new.scenario_name AND vbs_sbs.fold = stacking_new.fold AND vbs_sbs.metric = stacking_new.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    stacking_variance_threshold = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_new.approach, vbs_sbs.metric, stacking_new.result, ((stacking_new.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_new ON vbs_sbs.scenario_name = stacking_new.scenario_name AND vbs_sbs.fold = stacking_new.fold AND vbs_sbs.metric = stacking_new.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_VarianceThreshold' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    stacking_regression_new = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_new.approach, vbs_sbs.metric, stacking_new.result, ((stacking_new.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_new ON vbs_sbs.scenario_name = stacking_new.scenario_name AND vbs_sbs.fold = stacking_new.fold AND vbs_sbs.metric = stacking_new.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_SelectKBest_f_regression' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    stacking_mutual = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_new.approach, vbs_sbs.metric, stacking_new.result, ((stacking_new.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_new ON vbs_sbs.scenario_name = stacking_new.scenario_name AND vbs_sbs.fold = stacking_new.fold AND vbs_sbs.metric = stacking_new.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_SelectKBest_mutual_info_regression' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    stacking_after_fix = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_after_fix.approach, vbs_sbs.metric, stacking_after_fix.result, ((stacking_after_fix.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_after_fix ON vbs_sbs.scenario_name = stacking_after_fix.scenario_name AND vbs_sbs.fold = stacking_after_fix.fold AND vbs_sbs.metric = stacking_after_fix.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_per_algorithm_regressor' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    plt.axhline(np.average(stacking.result), color=color1, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(stacking_variance_threshold.result), color=color2, linestyle='dashed', linewidth=1)
    #plt.axhline(np.average(stacking_regression_new.result), color=color3, linestyle='dashed', linewidth=1)
    #plt.axhline(np.average(stacking_mutual.result), color=color4, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(stacking_after_fix.result), color=color5, linestyle='dashed', linewidth=1)

    width = 0.25  # the width of the bars
    ind = np.arange(len(stacking.scenario_name))
    ax.bar(ind - 3 * width / 2, stacking.result, width, color=color1, label='Stacking')
    ax.bar(ind - width / 2, stacking_variance_threshold.result, width, color=color2, label='Stacking Variance Threshold')
    #ax.bar(ind + width / 2, stacking_regression_new.result, width, color=color3,
    #       label='Stacking SelectKBest f_regression')
    #ax.bar(ind + 3 * width / 2, stacking_mutual.result, width, color=color4, label='Stacking SelectKBest mutual_info_regression')
    ax.bar(ind + width / 2, stacking_after_fix.result, width, color=color5,
           label='Stacking after fix')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(stacking.scenario_name)

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