import pandas as pd
import configparser
from matplotlib import pyplot as plt
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    voting_color = '#2a9d8f'
    stacking_color = '#e9c46a'
    run2survive_color = '#e76f51'
    bagging_color = '#264653'
    
    stacking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_after_fix .approach, vbs_sbs.metric, stacking_after_fix .result, ((stacking_after_fix .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_after_fix  ON vbs_sbs.scenario_name = stacking_after_fix .scenario_name AND vbs_sbs.fold = stacking_after_fix .fold AND vbs_sbs.metric = stacking_after_fix .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='stacking_per_algorithm_regressor' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_cross .approach, vbs_sbs.metric, voting_weighting_cross .result, ((voting_weighting_cross .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_cross  ON vbs_sbs.scenario_name = voting_weighting_cross .scenario_name AND vbs_sbs.fold = voting_weighting_cross .fold AND vbs_sbs.metric = voting_weighting_cross .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    run2survive = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='Expectation_algorithm_survival_forest' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    bagging = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, final_bagging_base_learner_test .approach, vbs_sbs.metric, final_bagging_base_learner_test .result, ((final_bagging_base_learner_test .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN final_bagging_base_learner_test  ON vbs_sbs.scenario_name = final_bagging_base_learner_test .scenario_name AND vbs_sbs.fold = final_bagging_base_learner_test .fold AND vbs_sbs.metric = final_bagging_base_learner_test .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_without_ranking' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    plt.axhline(np.average(stacking.result), color=stacking_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(voting.result), color=voting_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(run2survive.result), color=run2survive_color, linestyle='dashed', linewidth=1)
    plt.axhline(np.average(bagging.result), color=bagging_color, linestyle='dashed', linewidth=1)

    ind = np.arange(len(stacking.scenario_name))
    width = 0.2  # the width of the bars
    ax.bar(ind, stacking.result, width, color=stacking_color, label='Stacking')
    ax.bar(ind - width, voting.result, width, color=voting_color, label='Voting with cross validation')
    ax.bar(ind + width, run2survive.result, width, color=run2survive_color, label='Run2Survive')
    ax.bar(ind + 2 * width, bagging.result, width, color=bagging_color, label='Bagging')


    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(stacking.scenario_name)

    #ax.plot(stacking.scenario_name, stacking.result, color=stacking_color, label='Stacking')
    #ax.plot(voting.scenario_name, voting.result, color=voting_color, label='Voting')
    #ax.axis([0, 24, 0, 50])
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


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