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
    stacking_per_algorithm_regressor = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7per_algorithm_regressor' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_sunny = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_isac = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7ISAC' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_satzilla = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7SATzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_expectation = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7Expectation' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_par10 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7PAR10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_multiclass = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7multiclass' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_random_forest = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7RandomForest' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    stacking_svm = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_pred_only.approach, vbs_sbs.metric, stacking_pred_only.result, ((stacking_pred_only.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_pred_only ON vbs_sbs.scenario_name = stacking_pred_only.scenario_name AND vbs_sbs.fold = stacking_pred_only.fold AND vbs_sbs.metric = stacking_pred_only.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_1_2_3_4_5_6_7SVM' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")


    satzilla = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='satzilla-11' GROUP BY approach")

    fig = plt.figure(1, figsize=(7, 7))

    ax = fig.add_subplot(111)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking_per_algorithm_regressor.result, width, color=color1)
    ax.bar(2, stacking_sunny.result, width, color=color1)
    ax.bar(3, stacking_isac.result, width, color=color1)
    ax.bar(4, stacking_satzilla.result, width, color=color1)
    ax.bar(5, stacking_expectation.result, width, color=color3)
    ax.bar(6, stacking_par10.result, width, color=color1)
    ax.bar(7, stacking_multiclass.result, width, color=color1)
    ax.bar(8, stacking_random_forest.result, width, color=color1)
    ax.bar(9, stacking_svm.result, width, color=color1)

    plt.axhline(float(satzilla.result), color='#000', linestyle='dashed', linewidth=2)

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla", "SF-Exp.", "SF-PAR10", "Multi", "RFC", "SVM"])

    ax.text(1, float(stacking_per_algorithm_regressor.result), round(float(stacking_per_algorithm_regressor.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(stacking_sunny.result), round(float(stacking_sunny.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(stacking_isac.result), round(float(stacking_isac.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(stacking_satzilla.result) + 0.02, round(float(stacking_satzilla.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(5, float(stacking_satzilla.result) + 0.02, round(float(stacking_expectation.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(6, float(stacking_satzilla.result) + 0.02, round(float(stacking_par10.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(7, float(stacking_multiclass.result), round(float(stacking_multiclass.result) , 3), ha='center', va='bottom', rotation=0)
    ax.text(8, float(stacking_random_forest.result), round(float(stacking_random_forest.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(9, float(stacking_svm.result), round(float(stacking_svm.result), 3), ha='center', va='bottom', rotation=0)

    ax.text(10, float(satzilla.result) - 0.007, round(float(satzilla.result), 3), ha='center', va='bottom', rotation=0)

    #plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=0.35)
    ax.set_ylim(top=0.85)

    plt.title("f_stacking meta-learner comparison")
    plt.xlabel("Meta-learner")
    plt.ylabel("nPAR10")

    plt.show()

    fig.savefig("plotted/stacking_meta_learner.pdf", bbox_inches='tight')


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