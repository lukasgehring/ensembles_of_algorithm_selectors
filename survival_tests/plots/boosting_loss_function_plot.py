import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


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
    per_algorithm_regressor_l = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostr2_per_algorithm_regressor.approach, vbs_sbs.metric, adaboostr2_per_algorithm_regressor.result, ((adaboostr2_per_algorithm_regressor.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostr2_per_algorithm_regressor ON vbs_sbs.scenario_name = adaboostr2_per_algorithm_regressor.scenario_name AND vbs_sbs.fold = adaboostr2_per_algorithm_regressor.fold AND vbs_sbs.metric = adaboostr2_per_algorithm_regressor.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE 'adaboostR2_per_algorithm_regressor_linear___' OR approach LIKE 'adaboostR2_per_algorithm_regressor_linear__' GROUP BY scenario_name")
    per_algorithm_regressor_s = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostr2_per_algorithm_regressor.approach, vbs_sbs.metric, adaboostr2_per_algorithm_regressor.result, ((adaboostr2_per_algorithm_regressor.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostr2_per_algorithm_regressor ON vbs_sbs.scenario_name = adaboostr2_per_algorithm_regressor.scenario_name AND vbs_sbs.fold = adaboostr2_per_algorithm_regressor.fold AND vbs_sbs.metric = adaboostr2_per_algorithm_regressor.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE 'adaboostR2_per_algorithm_regressor_square___' OR approach LIKE 'adaboostR2_per_algorithm_regressor_square__' GROUP BY scenario_name")
    per_algorithm_regressor_e = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostr2_per_algorithm_regressor.approach, vbs_sbs.metric, adaboostr2_per_algorithm_regressor.result, ((adaboostr2_per_algorithm_regressor.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostr2_per_algorithm_regressor ON vbs_sbs.scenario_name = adaboostr2_per_algorithm_regressor.scenario_name AND vbs_sbs.fold = adaboostr2_per_algorithm_regressor.fold AND vbs_sbs.metric = adaboostr2_per_algorithm_regressor.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE 'adaboostR2_per_algorithm_regressor_exponential___' OR approach LIKE 'adaboostR2_per_algorithm_regressor_exponential___' GROUP BY scenario_name")
    per_algorithm_regressor = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='per_algorithm_RandomForestRegressor_regressor' GROUP BY scenario_name")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(10, 5))

    ax = fig.add_subplot(111)

    width = 0.2  # the width of the bars
    ind = np.arange(len(per_algorithm_regressor_l.result))
    ax.bar(ind - width, per_algorithm_regressor_l.result, width, color=color1, zorder=6)
    ax.bar(ind, per_algorithm_regressor_s.result, width, color=color2, zorder=6)
    ax.bar(ind + width, per_algorithm_regressor_e.result, width, color=color4, zorder=6)
    ax.plot(ind, per_algorithm_regressor.result, color=color3, zorder=6, marker='_', markersize=25, markeredgewidth=3, lw=0)
    plt.axhline(np.average(per_algorithm_regressor_l.result), color=color1, linestyle='dashed', linewidth=2)
    plt.axhline(np.average(per_algorithm_regressor_s.result), color=color2, linestyle='dashed', linewidth=2)
    plt.axhline(np.average(per_algorithm_regressor_e.result), color=color4, linestyle='dashed', linewidth=2)

    ax.set_xticks(ind)
    ax.set_xticklabels(
        ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18",
         "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16\_INDU", "SAT12-INDU", "SAT18-EXP"], fontsize=7)

    plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('nPAR10', fontsize=11)
    #ax.set_xlabel('Scenario', fontsize=11)

    # ax.set_ylim(bottom=0.3)
    # ax.set_ylim(top=0.7)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    l1 = mpatches.Patch(color=color1, label="linear loss")
    l2 = mpatches.Patch(color=color2, label="square loss")
    l3 = mpatches.Patch(color=color4, label="exponential loss")
    l4 = mpatches.Patch(color=color3, label="PerAlgorithmRegressor")

    fig.legend(handles=[l1, l2, l3, l4], loc=2, bbox_to_anchor=(0.06, 0.98), prop={'size': 9})
    plt.legend()
    plt.show()

    fig.savefig("plotted/boostingR2_loss.pdf", bbox_inches='tight')


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