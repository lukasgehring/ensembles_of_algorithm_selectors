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
    
    stacking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking .approach, vbs_sbs.metric, stacking .result, ((stacking .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking  ON vbs_sbs.scenario_name = stacking .scenario_name AND vbs_sbs.fold = stacking .fold AND vbs_sbs.metric = stacking .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_2_4_5_6SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting_weighting_ranking_24567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_weighting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    samme = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme_mulitclass.approach, vbs_sbs.metric, adaboostsamme_mulitclass.result, ((adaboostsamme_mulitclass.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme_mulitclass ON vbs_sbs.scenario_name = adaboostsamme_mulitclass.scenario_name AND vbs_sbs.fold = adaboostsamme_mulitclass.fold AND vbs_sbs.metric = adaboostsamme_mulitclass.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_multiclass_algorithm_selector_30' GROUP BY scenario_name")
    bagging_40 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_40_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY scenario_name")

    sunny = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='sunny' GROUP BY scenario_name")
    satzilla = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='satzilla-11' GROUP BY scenario_name")
    exp = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='Expectation_algorithm_survival_forest' GROUP BY scenario_name")
    multi = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='multiclass_algorithm_selector' GROUP BY scenario_name")

    best_baseline = [sunny.result[0], satzilla.result[1], satzilla.result[2], satzilla.result[3], satzilla.result[4], exp.result[5], multi.result[6], satzilla.result[7], sunny.result[8], satzilla.result[9], exp.result[10], satzilla.result[11], exp.result[12]]
    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(10, 5))

    ax1 = fig.add_subplot(111)

    #ax1.text(13.8, np.average(stacking.result), round(np.average(stacking.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(stacking.result), color=color1, linestyle='dashed', linewidth=1.4)

    #ax1.text(13.8, np.average(samme.result), round(np.average(samme.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(samme.result), color=color3, linestyle='dashed', linewidth=1.4)

    #ax1.text(13.8, np.average(bagging_40.result), round(np.average(bagging_40.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(bagging_40.result), color=color4, linestyle='dashed', linewidth=1.4)

    #ax1.text(13.8, np.average(voting_weighting_ranking_24567.result), round(np.average(voting_weighting_ranking_24567.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(voting_weighting_ranking_24567.result), color=color2, linestyle='dashed', linewidth=1.4)

    ind = np.arange(len(stacking.scenario_name))
    width = 0.15  # the width of the bars
    ax1.bar(ind, stacking.result, width, color=color1, label='Stacking', zorder=6)
    ax1.bar(ind - width, voting_weighting_ranking_24567.result, width, color=color2, label='Voting', zorder=6)
    ax1.bar(ind + width, samme.result, width, color=color3, label='Boosting', zorder=6)
    ax1.bar(ind - 2 * width, bagging_40.result, width, color=color4, label='Bagging', zorder=6)
    ax1.bar(ind + 2 * width, best_baseline, width, color=color5, label='Best Single Learner', zorder=6)


    ax1.set_xticks(ind)
    ax1.set_xticklabels(
        ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18",
         "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16\_INDU", "SAT12-INDU", "SAT18-EXP"], fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    ax1.set_ylabel('nPAR10', fontsize=11)

    plt.legend()
    plt.show()

    fig.savefig("plotted/comparison_plot.pdf", bbox_inches='tight')


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