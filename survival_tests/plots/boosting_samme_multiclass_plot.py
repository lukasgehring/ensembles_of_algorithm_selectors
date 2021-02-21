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
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='ASP-POTASSCO'")
    bnsl = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='BNSL-2016'")
    cpmp = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CPMP-2015'")
    csp = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CSP-2010'")
    csp_time = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CSP-Minizinc-Time-2016'")
    csp_mzn = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='CSP-MZN-2013'")
    gluhack = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='GLUHACK-18'")
    maxsat12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='MAXSAT12-PMS'")
    maxsat15 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='MAXSAT15-PMS-INDU'")
    qbf = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='QBF-2011'")
    sat03 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='SAT03-16_INDU'")
    sat12 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='SAT12-INDU'")
    sat18 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, samme_multiclass_tesst.approach, vbs_sbs.metric, samme_multiclass_tesst.result, ((samme_multiclass_tesst.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN samme_multiclass_tesst ON vbs_sbs.scenario_name = samme_multiclass_tesst.scenario_name AND vbs_sbs.fold = samme_multiclass_tesst.fold AND vbs_sbs.metric = samme_multiclass_tesst.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%SAMME_multiclass_algorithm_selector%' AND scenario_name='SAT18-EXP'")

    base_learner = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%multiclass%' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    max_it = 200
    dfs = [asp, bnsl, cpmp, csp, csp_time, csp_mzn, gluhack, maxsat12, maxsat15, qbf, sat03, sat12, sat18]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #print(multiclass_algorithm_selector)

    average_performance = np.zeros(max_it)

    fig = plt.figure(1, figsize=(20, 8))

    for i, df in enumerate(dfs):
        if df.empty:
            continue
        if i >= 10:
            pos = i + 1
        else:
            pos = i
        ax1 = fig.add_subplot(3, 5, pos + 1)

        ax1.axhline(np.average(base_learner.result[i]), color=color1, linestyle='dashed', linewidth=1.4)
        print(base_learner.scenario_name[i])
        # code by https://stackoverflow.com/questions/23493374/sort-dataframe-index-that-has-a-string-and-number
        # ----------------------
        df['indexNumber'] = [int(i.split('_')[-1]) for i in df.approach]
        df.sort_values(['indexNumber', 'fold'], ascending=[True, True], inplace=True)
        df.drop('indexNumber', 1, inplace=True)
        # ----------------------

        best_data = {}
        plot_data = []
        for iter in range(1, max_it + 1):
            data = []
            for fold in range(1, 11):
                approach = 'SAMME_multiclass_algorithm_selector_%d' % (iter)
                val = df.loc[(df['approach'] == approach) & (df['fold'] == fold)].result
                if len(val) == 1:
                    key = str(fold)
                    best_data[key] = val.iloc[0]
                    data.append(best_data[key])
                else:
                    data.append(best_data[str(fold)])
            plot_data.append(np.average(data))

        ax1.plot(range(1, max_it + 1), plot_data)
        average_performance = average_performance + plot_data

        # ax1.plot(range(1, len(sorted_data) + 1), sorted_data)
        plt.xlabel('Iterations')
        plt.ylabel('nPAR10')

        plt.xlim((1, max_it + 1))

        plt.title(df.scenario_name[0])

    plt.show()

    average_performance = average_performance / 13

    print(average_performance.tolist())

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