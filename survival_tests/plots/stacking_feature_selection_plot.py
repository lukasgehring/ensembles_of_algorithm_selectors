import pandas as pd
import configparser
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
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

    stacking_feature_selection = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as counter FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_feature_selection.approach, vbs_sbs.metric, stacking_feature_selection.result, ((stacking_feature_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_feature_selection ON vbs_sbs.scenario_name = stacking_feature_selection.scenario_name AND vbs_sbs.fold = stacking_feature_selection.fold AND vbs_sbs.metric = stacking_feature_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND NOT approach LIKE '%RandomForest%' GROUP BY approach ORDER BY approach")
    stacking = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as counter FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%1_2_3_4_5_6_7%' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND NOT approach LIKE '%RandomForest%' GROUP BY approach ORDER BY approach")

    #plt.rc('font', family='sans-serif')
    #plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(10, 5))

    ax = fig.add_subplot(111)

    width = 0.25  # the width of the bars
    ind = np.arange(len(stacking_feature_selection.result))
    index = 0
    permutation = [4, 2, 6, 5, 0, 3, 1, 7]
    for i in ind:

        if i % 2 == 0:
            print(stacking_feature_selection.approach[i])
            ax.bar(permutation[index], stacking_feature_selection.result[i], width, color=color1, zorder=6)
        else:
            ax.bar(permutation[index] + width, stacking_feature_selection.result[i], width, color=color3, zorder=6)
            index = index + 1
    for i in range(index):
        ax.bar(permutation[i] - width, stacking.result[i*2], width, color=color2, zorder=6)

    ax.set_xticks(range(len(permutation)))
    ax.set_xticklabels(ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla'11", "SF-Exp.", "SF-PAR10", "Multiclass", "SVM"]))


    #plt.xticks(rotation=90)

    ax.set_ylabel('nPAR10', fontsize=11)
    ax.set_xlabel('Meta-Learner', fontsize=11)

    ax.set_ylim(bottom=0.3)
    #ax.set_ylim(top=0.7)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    l1 = mpatches.Patch(color=color1, label="Stacking with SelectKBest")
    l2 = mpatches.Patch(color=color2, label="Stacking without feature selection")
    l3 = mpatches.Patch(color=color3, label="Stacking with VarianceThreshold")

    fig.legend(handles=[l2, l1, l3], loc=1, prop={'size': 13}, bbox_to_anchor=(0.99, 0.98))

    plt.show()

    fig.savefig("plotted/stacking_feature_selection.pdf", bbox_inches='tight')


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