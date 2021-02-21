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
    per_algorithm_regressor_base_learner = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='per_algorithm_RandomForestRegressor_regressor' GROUP BY scenario_name")
    sunny_base_learner = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SUNNY' GROUP BY scenario_name")
    multiclass_base_learner = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='multiclass_algorithm_selector' GROUP BY approach")


    per_algorithm_regressor = 0.3955381954847779
    multiclass_algorithm_selector = 0.43744382259771625
    sunny = 0.49750856758840417
    fig = plt.figure(1, figsize=(6, 4))

    ax = fig.add_subplot(111)

    a = list()
    a.append(per_algorithm_regressor)
    a.append(multiclass_algorithm_selector)
    a.append(sunny)

    c = list()
    c.append(np.average(per_algorithm_regressor_base_learner.result))
    c.append(np.average(multiclass_base_learner.result))
    c.append(np.average(sunny_base_learner.result))

    b = list()
    b.append("PerAlgo")
    b.append("Multiclass")
    b.append("SUNNY")

    width = 0.2
    # the width of the bars
    ind = np.arange(len(b))


    ax.bar(ind + width / 2, c, width, color=color1, label='Single Learner', zorder=6)
    ax.bar(ind - width / 2, a, width, color=color2, label='Boosted Learner', zorder=6)

    ax.text(0 + width / 2, c[0], round(c[0], 3), ha='center', va='bottom', rotation=0)
    ax.text(1 + width / 2, c[1], round(c[1], 3), ha='center', va='bottom', rotation=0)
    ax.text(2 + width / 2, c[2], round(c[2], 3), ha='center', va='bottom', rotation=0)

    ax.text(0 - width / 2, a[0], round(a[0], 3), ha='center', va='bottom', rotation=0)
    ax.text(1 - width / 2, a[1], round(a[1], 3), ha='center', va='bottom', rotation=0)
    ax.text(2 - width / 2, a[2], round(a[2], 3), ha='center', va='bottom', rotation=0)

    ax.set_xticks(ind)
    ax.set_xticklabels(b)

    plt.xlabel('Learning Algorithm')
    plt.ylabel('nPAR10')

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    # 100 linearly spaced numbers
    #x = np.linspace(0, 14, 15)

    # approximation plot for PerAlgorithmRegressor
    #y = 0.0007905*(x** 2) + (-0.01650208)*x + 0.47456491
    #ax.plot(x, y, 'r', color= color1, linewidth=2, label="PerAlgorithmRegressor approximation")

    # approximation plot for SUNNY
    #y = 0.0009834 * (x ** 2) + (-0.018609) * x + 0.46738839
    #ax.plot(x, y, 'r', color= color2, linewidth=2, label="SUNNY approximation")

    ax.set_ylim(bottom=0.3)
    ax.set_ylim(top=0.7)

    plt.legend()

    plt.show()

    fig.savefig("plotted/samme_base_learner.pdf", bbox_inches='tight')


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