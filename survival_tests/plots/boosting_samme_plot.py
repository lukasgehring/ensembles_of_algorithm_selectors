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
    per_algorithm_regressor = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='per_algorithm_RandomForestRegressor_regressor' GROUP BY scenario_name")
    boosting_1 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_1' GROUP BY scenario_name")
    boosting_2 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_2' GROUP BY scenario_name")
    boosting_3 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_3' GROUP BY scenario_name")
    boosting_4 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_4' GROUP BY scenario_name")
    boosting_5 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_5' GROUP BY scenario_name")
    boosting_6 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_6' GROUP BY scenario_name")
    boosting_7 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_7' GROUP BY scenario_name")
    boosting_8 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_8' GROUP BY scenario_name")
    boosting_9 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_9' GROUP BY scenario_name")
    boosting_10 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme.approach, vbs_sbs.metric, adaboostsamme.result, ((adaboostsamme.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme ON vbs_sbs.scenario_name = adaboostsamme.scenario_name AND vbs_sbs.fold = adaboostsamme.fold AND vbs_sbs.metric = adaboostsamme.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_per_algorithm_regressor_10' GROUP BY scenario_name")


    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    # PerAlgorithmRegressor
    frames = [boosting_1.result, boosting_2.result, boosting_3.result, boosting_4.result, boosting_5.result, boosting_6.result,
              boosting_7.result, boosting_8.result, boosting_9.result, boosting_10.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(2).T
    table = result.to_latex()
    table = table.replace("\n0  ", "\nASP-POTASSCO          ")
    table = table.replace("\n1  ", "\nBNSL-2016             ")
    table = table.replace("\n2  ", "\nCPMP-2015             ")
    table = table.replace("\n3  ", "\nCSP-2010              ")
    table = table.replace("\n4  ", "\nCSP-Minizinc-Time-2016")
    table = table.replace("\n5  ", "\nCSP-MZN-2013          ")
    table = table.replace("\n6  ", "\nGLUHACK-18            ")
    table = table.replace("\n7  ", "\nMAXSAT12-PMS          ")
    table = table.replace("\n8  ", "\nMAXSAT15-PMS-INDU     ")
    table = table.replace("\n9  ", "\nQBF-2011              ")
    table = table.replace("\n10 ", "\nSAT03-16\_INDU        ")
    table = table.replace("\n11 ", "\nSAT12-INDU            ")
    table = table.replace("\n12 ", "\nSAT18-EXP             ")
    print(table)

    # SUNNY
    #frames = [SUNNY_4.result, SUNNY_8.result, SUNNY_12.result, SUNNY_16.result, SUNNY_20.result,
    #          SUNNY_24.result, SUNNY_28.result, SUNNY_32.result, SUNNY_36.result, SUNNY_40.result,
    #          SUNNY_44.result, SUNNY_48.result, SUNNY_52.result, SUNNY_56.result, SUNNY_60.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(2).T
    table = result.to_latex()
    table.replace("0 ", "ASP-POTASSCO ")
    table.replace("1 ", "BNSL-2016 ")
    table.replace("2 ", "CPMP-2015 ")
    table.replace("3 ", "CSP-2010 ")
    table.replace("4 ", "CSP-Minizinc-Time-2016 ")
    table.replace("5 ", "CSP-MZN-2013 ")
    table.replace("6 ", "GLUHACK-18 ")
    table.replace("7 ", "MAXSAT12-PMS ")
    table.replace("8 ", "MAXSAT15-PMS-INDU ")
    table.replace("9 ", "QBF-2011 ")
    table.replace("10 ", "SAT03-16\_INDU ")
    table.replace("11 ", "SAT12-INDU ")
    table.replace("12 ", "SAT18-EXP ")
    print(table)


    a = list()
    a.append(np.average(boosting_1.result))
    a.append(np.average(boosting_2.result))
    a.append(np.average(boosting_3.result))
    a.append(np.average(boosting_4.result))
    a.append(np.average(boosting_5.result))
    a.append(np.average(boosting_6.result))
    a.append(np.average(boosting_7.result))
    a.append(np.average(boosting_8.result))
    a.append(np.average(boosting_9.result))
    a.append(np.average(boosting_10.result))

    # c = list()
    # c.append(np.average(SUNNY_4.result))
    # c.append(np.average(SUNNY_8.result))
    # c.append(np.average(SUNNY_12.result))
    # c.append(np.average(SUNNY_16.result))
    # c.append(np.average(SUNNY_20.result))
    # c.append(np.average(SUNNY_24.result))
    # c.append(np.average(SUNNY_28.result))
    # c.append(np.average(SUNNY_32.result))
    # c.append(np.average(SUNNY_36.result))
    # c.append(np.average(SUNNY_40.result))
    # c.append(np.average(SUNNY_44.result))
    # c.append(np.average(SUNNY_48.result))
    # c.append(np.average(SUNNY_52.result))
    # c.append(np.average(SUNNY_56.result))
    # c.append(np.average(SUNNY_60.result))

    b = list()
    b.append("1")
    b.append("2")
    b.append("3")
    b.append("4")
    b.append("5")
    b.append("6")
    b.append("7")
    b.append("8")
    b.append("9")
    b.append("10")

    width = 0.3
    # the width of the bars
    ind = np.arange(len(b))


    ax.bar(ind + width / 2, a, width, color=color1, label='PerAlgorithmRegressor')
    #ax.bar(ind - width / 2, c, width, color=color2, label='SUNNY')

    ax.set_xticks(ind)
    ax.set_xticklabels(b)

    plt.xlabel('Number of Base Learner')
    plt.ylabel('nPAR10')

    # 100 linearly spaced numbers
    #x = np.linspace(0, 14, 15)

    # approximation plot for PerAlgorithmRegressor
    #y = 0.0007905*(x** 2) + (-0.01650208)*x + 0.47456491
    #ax.plot(x, y, 'r', color= color1, linewidth=2, label="PerAlgorithmRegressor approximation")

    # approximation plot for SUNNY
    #y = 0.0009834 * (x ** 2) + (-0.018609) * x + 0.46738839
    #ax.plot(x, y, 'r', color= color2, linewidth=2, label="SUNNY approximation")

    ax.set_ylim(bottom=0.0)
    ax.set_ylim(top=0.7)

    plt.xticks(rotation=90)

    plt.legend()

    plt.show()

    #fig.savefig("foo.pdf", bbox_inches='tight')


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