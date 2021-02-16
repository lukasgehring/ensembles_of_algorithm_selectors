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

    # PerAlgorithmRegressor
    asp = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='ASP-POTASSCO' GROUP BY scenario_name, approach")
    bnsl = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='BNSL-2016' GROUP BY scenario_name, approach")
    cpmp = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CPMP-2015' GROUP BY scenario_name, approach")
    csp = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CSP-2010' GROUP BY scenario_name, approach")
    csp_time = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CSP-Minizinc-Time-2016' GROUP BY scenario_name, approach")
    csp_mzn = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CSP-MZN-2013' GROUP BY scenario_name, approach")
    gluhack = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='GLUHACK-18' GROUP BY scenario_name, approach")
    maxsat12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='MAXSAT12-PMS' GROUP BY scenario_name, approach")
    maxsat15 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='MAXSAT15-PMS-INDU' GROUP BY scenario_name, approach")
    qbf = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='QBF-2011' GROUP BY scenario_name, approach")
    sat03 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='SAT03-16_INDU' GROUP BY scenario_name, approach")
    sat12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='SAT12-INDU' GROUP BY scenario_name, approach")
    sat18 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, COUNT(fold) as folds, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='SAT18-EXP' GROUP BY scenario_name, approach")

    multiclass = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%multiclass%' GROUP BY scenario_name")

    dfs = [asp, bnsl, cpmp, csp, csp_time, csp_mzn, gluhack, maxsat12, maxsat15, qbf, sat03, sat12, sat18]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #print(multiclass_algorithm_selector)

    #print(dfs)

    fig = plt.figure(1, figsize=(20, 8))

    for i, df in enumerate(dfs):
        if df.empty:
            continue
        if i >= 10:
            pos = i + 1
        else:
            pos = i
        ax1 = fig.add_subplot(3, 5, pos + 1)

        data1 = list()
        data2 = list()
        ticks = list()

        for approach, result, folds in zip(df.approach, df.result, df.folds):
            ticks.append(int(approach.replace('SAMME_multiclass_algorithm_selector_', '')))
            data1.append(result)
            data2.append(folds)
        sorted_data = [x for _, x in sorted(zip(ticks, data1))]
        ax1.plot(range(1, len(sorted_data) + 1), sorted_data)
        plt.xlabel('Iterations')
        plt.ylabel('nPAR10')
        ax1.axhline(multiclass.result[i], color='#000', linestyle='dashed', linewidth=2)


        ax2 = ax1.twinx()
        sorted_data = [x for _, x in sorted(zip(ticks, data2))]
        ax2.plot(range(1, len(sorted_data) + 1), sorted_data, color=color3)
        ax2.set_yticks([0, 2, 4, 6, 8, 10])
        ax2.tick_params(axis='y', colors=color3)
        plt.ylabel('folds running', color=color3)

        plt.xlim((1, len(sorted_data)))

        plt.title(df.scenario_name[0])


    plt.show()

    fig.savefig("plotted/samme_multiclass.pdf", bbox_inches='tight')


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