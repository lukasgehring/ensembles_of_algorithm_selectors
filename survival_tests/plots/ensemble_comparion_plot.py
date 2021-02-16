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
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking .approach, vbs_sbs.metric, stacking .result, ((stacking .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking  ON vbs_sbs.scenario_name = stacking .scenario_name AND vbs_sbs.fold = stacking .fold AND vbs_sbs.metric = stacking .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_2_4_5_6SUNNY' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_weighting_ranking_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_weighting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    samme = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme_mulitclass.approach, vbs_sbs.metric, adaboostsamme_mulitclass.result, ((adaboostsamme_mulitclass.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme_mulitclass ON vbs_sbs.scenario_name = adaboostsamme_mulitclass.scenario_name AND vbs_sbs.fold = adaboostsamme_mulitclass.fold AND vbs_sbs.metric = adaboostsamme_mulitclass.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='SAMME_multiclass_algorithm_selector_30' GROUP BY approach")
    bagging_40 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_number_of_base_learner.approach, vbs_sbs.metric, bagging_number_of_base_learner.result, ((bagging_number_of_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_number_of_base_learner ON vbs_sbs.scenario_name = bagging_number_of_base_learner.scenario_name AND vbs_sbs.fold = bagging_number_of_base_learner.fold AND vbs_sbs.metric = bagging_number_of_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='bagging_40_per_algorithm_RandomForestRegressor_regressor_without_ranking' GROUP BY approach")
    satzilla = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='satzilla-11' GROUP BY approach")

    stacking_unsolved_instances = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM `stacking` WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_4_5_6SUNNY' GROUP BY approach")
    voting_unsolved_instances = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM `voting_ranking` WHERE metric='number_unsolved_instances_False' AND approach='voting_ranking_weighting_2_4_5_6_7' GROUP BY approach")
    samme_unsolved_instances = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM `adaboostsamme_mulitclass` WHERE metric='number_unsolved_instances_False' AND approach='SAMME_multiclass_algorithm_selector_30' GROUP BY approach")
    bagging_unsolved_instances = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM `bagging_number_of_base_learner` WHERE metric='number_unsolved_instances_False' AND approach='bagging_40_per_algorithm_RandomForestRegressor_regressor_without_ranking' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    satzilla_unsolved_instances = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM `pre_computed_base_learner` WHERE metric='number_unsolved_instances_False' AND approach='satzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(10, 4))

    ax1 = fig.add_subplot(111)

    ind = np.arange(5)
    width = 0.5  # the width of the bars
    ax1.bar(4, satzilla.result, width, color=color2, label='Stacking', zorder=6)
    ax1.bar(0, stacking.result, width, color=color1, label='Stacking', zorder=6)
    ax1.bar(1, voting_weighting_ranking_24567.result, width, color=color1, label='Voting', zorder=6)
    ax1.bar(2, samme.result, width, color=color1, label='Boosting', zorder=6)
    ax1.bar(3, bagging_40.result, width, color=color1, label='Bagging', zorder=6)

    ax1.text(0, float(stacking.result), round(float(stacking.result), 3), ha='center', va='bottom', rotation=0)
    ax1.text(1, float(voting_weighting_ranking_24567.result), round(float(voting_weighting_ranking_24567.result), 3), ha='center', va='bottom', rotation=0)
    ax1.text(2, float(samme.result), round(float(samme.result), 3), ha='center', va='bottom', rotation=0)
    ax1.text(3, float(bagging_40.result), round(float(bagging_40.result), 3), ha='center', va='bottom', rotation=0)
    ax1.text(4, float(satzilla.result), round(float(satzilla.result), 3), ha='center', va='bottom', rotation=0)

    ax1.set_xticks(ind)
    #plt.xticks(rotation=45, ha='right')
    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    ax1.set_ylabel('nPAR10', fontsize=11)
    #ax1.set_xlabel('Approach', fontsize=11)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('unsolved instances (\%)')
    ax2.plot(0, stacking_unsolved_instances.unsolved_instances * 100, marker='s', markersize=8, lw=0, color=color3)
    ax2.plot(1, voting_unsolved_instances.unsolved_instances * 100, marker='s', markersize=8, lw=0, color=color3)
    ax2.plot(2, samme_unsolved_instances.unsolved_instances * 100, marker='s', markersize=8, lw=0, color=color3)
    ax2.plot(3, bagging_unsolved_instances.unsolved_instances * 100, marker='s', markersize=8, lw=0, color=color3)
    ax2.plot(4, satzilla_unsolved_instances.unsolved_instances * 100, marker='s', markersize=8, lw=0, color=color3)
    ax2.tick_params(axis='y', colors=color3)

    ax1.set_ylim(bottom=0.2)
    ax1.set_ylim(top=0.5)

    ax2.set_ylim(bottom=0.04 * 100)
    ax2.set_ylim(top=0.1 * 100)

    ax1.set_xticklabels(["Stacking", "Voting", "Boosting", "Bagging", "SATzilla'11"])

    #plt.legend()
    plt.show()

    fig.savefig("plotted/ensemble_plot.pdf", bbox_inches='tight')


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