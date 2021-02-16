import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    color2 = '#264653'
    color1 = '#2a9d8f'

    bagging_per_algorithm_regressor= get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY approach")
    bagging_satzilla = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_satzilla-11_without_ranking' GROUP BY approach")
    bagging_expectation = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_Expectation_algorithm_survival_forest_without_ranking' GROUP BY approach")
    bagging_sunny = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_sunny_without_ranking' GROUP BY approach")
    bagging_isac = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_isac_without_ranking' GROUP BY approach")
    bagging_multiclass = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test.approach, vbs_sbs.metric, final_bagging_base_learner_test.result, ((final_bagging_base_learner_test.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test ON vbs_sbs.scenario_name = final_bagging_base_learner_test.scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test.fold AND vbs_sbs.metric = final_bagging_base_learner_test.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_10_multiclass_algorithm_selector_without_ranking' GROUP BY approach")

    per_algorithm_regressor = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='per_algorithm_RandomForestRegressor_regressor' GROUP BY approach")
    satzilla = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='satzilla-11' GROUP BY approach")
    expectation = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='Expectation_algorithm_survival_forest' GROUP BY approach")
    sunny = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='sunny' GROUP BY approach")
    isac = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='isac' GROUP BY approach")
    multiclass = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='multiclass_algorithm_selector' GROUP BY approach")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(8, 4))

    ax = fig.add_subplot(111)

    bagging = list()
    bagging.append(float(bagging_per_algorithm_regressor.result))
    bagging.append(float(bagging_satzilla.result))
    bagging.append(float(bagging_expectation.result))
    bagging.append(float(bagging_sunny.result))
    bagging.append(float(bagging_isac.result))
    bagging.append(float(bagging_multiclass.result))

    base_learner = list()
    base_learner.append(float(per_algorithm_regressor.result))
    base_learner.append(float(satzilla.result))
    base_learner.append(float(expectation.result))
    base_learner.append(float(sunny.result))
    base_learner.append(float(isac.result))
    base_learner.append(float(multiclass.result))

    names = list()
    names.append("PerAlgo")
    names.append("SatZilla")
    names.append("SF-Exp")
    names.append("SUNNY")
    names.append("ISAC")
    names.append("Multiclass")

    width = 0.25  # the width of the bars
    ind = np.arange(len(names))
    ax.bar(ind + width, bagging, width, color=color1,
           label='Bagging with 10 base learners', zorder=3)
    ax.bar(ind, base_learner, width, color=color2, label='Single Learner', zorder=3)

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(names)

    ax.set_yticks([0,0.2,0.4,0.6,0.8])

    plt.xlabel('Learing Algorithm')
    plt.ylabel('nPAR10')

    ax.text(0 + width, bagging[0], round(bagging[0], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(1 + width, bagging[1], round(bagging[1], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(2 + width, bagging[2], round(bagging[2], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(3 + width, bagging[3], round(bagging[3], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(4 + width, bagging[4], round(bagging[4], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(5 + width, bagging[5], round(bagging[5], 3), ha='center', va='bottom', rotation=0, fontsize=7)

    ax.text(0, base_learner[0], round(base_learner[0], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(1, base_learner[1], round(base_learner[1], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(2, base_learner[2], round(base_learner[2], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(3, base_learner[3], round(base_learner[3], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(4, base_learner[4], round(base_learner[4], 3), ha='center', va='bottom', rotation=0, fontsize=7)
    ax.text(5, base_learner[5], round(base_learner[5], 3), ha='center', va='bottom', rotation=0, fontsize=7)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    #plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=0)

    plt.legend()
    plt.show()

    fig.savefig("plotted/bagging_base_learner.pdf", bbox_inches='tight')


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