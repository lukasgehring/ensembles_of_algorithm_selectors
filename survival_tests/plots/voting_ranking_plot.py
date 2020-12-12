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
    voting_rank1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_1_2_3_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_rank24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_2_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_rank_max1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_max_1_2_3_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_rank_max24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_max_2_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_rank_min1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_min_1_2_3_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting_rank_min24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_ranking_min_2_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")


    voting1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting.approach, vbs_sbs.metric, voting.result, ((voting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting ON vbs_sbs.scenario_name = voting.scenario_name AND vbs_sbs.fold = voting.fold AND vbs_sbs.metric = voting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")
    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting.approach, vbs_sbs.metric, voting.result, ((voting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting ON vbs_sbs.scenario_name = voting.scenario_name AND vbs_sbs.fold = voting.fold AND vbs_sbs.metric = voting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_2_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.

    width = 0.2  # the width of the bars

    ax.bar(0.7, voting1234567.result, width, color=color1, label='without ranking')
    ax.bar(0.9, voting_rank1234567.result, width, color=color2, label='with avg ranking')
    ax.bar(1.1, voting_rank_max1234567.result, width, color=color3, label='with max ranking')
    ax.bar(1.3, voting_rank_min1234567.result, width, color=color4, label='with min ranking')

    ax.bar(1.7, voting24567.result, width, color=color1)
    ax.bar(1.9, voting_rank24567.result, width, color=color2)
    ax.bar(2.1, voting_rank_max24567.result, width, color=color3)
    ax.bar(2.3, voting_rank_min24567.result, width, color=color4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Full Voting", "Selected Voting"])

    #ax.text(1, float(voting_rank.result), round(float(voting_rank.result), 3), ha='center', va='bottom', rotation=0)

    ax.text(0.7, float(voting1234567.result), round(float(voting1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(0.9, float(voting_rank1234567.result), round(float(voting_rank1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.1, float(voting_rank_max1234567.result), round(float(voting_rank_max1234567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.3, float(voting_rank_min1234567.result), round(float(voting_rank_min1234567.result), 3), ha='center', va='bottom', rotation=0)

    ax.text(1.7, float(voting24567.result), round(float(voting24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(1.9, float(voting_rank24567.result), round(float(voting_rank24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2.1, float(voting_rank_max24567.result), round(float(voting_rank_max24567.result), 3), ha='center', va='bottom', rotation=0)
    ax.text(2.3, float(voting_rank_min24567.result), round(float(voting_rank_min24567.result), 3), ha='center', va='bottom', rotation=0)

    #plt.xticks(rotation=90)

    ax.set_ylim(bottom=0.36)
    ax.set_ylim(top=0.44)

    ax.set_ylabel('nPAR10')

    plt.legend()

    plt.show()

    fig.savefig("ranking.pdf", bbox_inches='tight')



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