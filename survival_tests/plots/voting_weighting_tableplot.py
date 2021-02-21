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

    # Voting
    voting1234567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection .approach, vbs_sbs.metric, voting_base_learner_selection .result, ((voting_base_learner_selection .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection  ON vbs_sbs.scenario_name = voting_base_learner_selection .scenario_name AND vbs_sbs.fold = voting_base_learner_selection .fold AND vbs_sbs.metric = voting_base_learner_selection .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting_weighting_1234567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_1_2_3_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    # Selected Voting
    voting24567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_base_learner_selection .approach, vbs_sbs.metric, voting_base_learner_selection .result, ((voting_base_learner_selection .result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_base_learner_selection  ON vbs_sbs.scenario_name = voting_base_learner_selection .scenario_name AND vbs_sbs.fold = voting_base_learner_selection .fold AND vbs_sbs.metric = voting_base_learner_selection .metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")
    voting_weighting_24567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting_selection.approach, vbs_sbs.metric, voting_weighting_selection.result, ((voting_weighting_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting_selection ON vbs_sbs.scenario_name = voting_weighting_selection.scenario_name AND vbs_sbs.fold = voting_weighting_selection.fold AND vbs_sbs.metric = voting_weighting_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_weighting_2_4_5_6_7' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")


    # Voting
    frames = [voting1234567.result, voting_weighting_1234567.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(3).T
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

    # Selected Voting
    frames = [voting24567.result, voting_weighting_24567.result]
    result = pd.concat(frames, axis=1).T
    result['average'] = result.mean(numeric_only=True, axis=1)
    result['median'] = result.median(numeric_only=True, axis=1)
    result = result.round(3).T
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

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(10, 5))

    ax1 = fig.add_subplot(111)

    # ax1.text(13.8, np.average(stacking.result), round(np.average(stacking.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(voting1234567.result), color=color1, linestyle='dashed', linewidth=1.4)

    # ax1.text(13.8, np.average(samme.result), round(np.average(samme.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(voting_weighting_1234567.result), color=color2, linestyle='dashed', linewidth=1.4)

    # ax1.text(13.8, np.average(bagging_40.result), round(np.average(bagging_40.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(voting24567.result), color=color3, linestyle='dashed', linewidth=1.4)

    # ax1.text(13.8, np.average(voting_weighting_ranking_24567.result), round(np.average(voting_weighting_ranking_24567.result), 2), ha='center', va='bottom', rotation=0, zorder=6)
    plt.axhline(np.average(voting_weighting_24567.result), color=color4, linestyle='dashed', linewidth=1.4)

    ind = np.arange(len(voting1234567.scenario_name))
    width = 0.2  # the width of the bars
    ax1.bar(ind - width / 2 - width, voting1234567.result, width, color=color1, label='Voting Ensemble with Majority Voting', zorder=6)
    ax1.bar(ind - width / 2, voting_weighting_1234567.result, width, color=color2, label='Voting Ensemble with Weighted Voting', zorder=6)
    ax1.bar(ind + width / 2, voting24567.result, width, color=color3, label='Selected Voting Ensemble with Majority Voting', zorder=6)
    ax1.bar(ind + width / 2 + width, voting_weighting_24567.result, width, color=color4, label='Selected Voting Ensemble with Weighted Voting', zorder=6)

    ax1.set_xticks(ind)
    ax1.set_xticklabels(
        ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18",
         "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16\_INDU", "SAT12-INDU", "SAT18-EXP"], fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    ax1.set_ylabel('nPAR10', fontsize=11)

    plt.legend()
    plt.show()

    fig.savefig("plotted/voting_weighting_comparison_plot.pdf", bbox_inches='tight')


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