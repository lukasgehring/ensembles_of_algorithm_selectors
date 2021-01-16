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

    voting1234567 = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting.approach, vbs_sbs.metric, voting.result, ((voting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting ON vbs_sbs.scenario_name = voting.scenario_name AND vbs_sbs.fold = voting.fold AND vbs_sbs.metric = voting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='voting_1_2_3_4_5_6_7_pre' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY scenario_name")

    satzilla = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, base_learner.approach, vbs_sbs.metric, base_learner.result, ((base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN base_learner ON vbs_sbs.scenario_name = base_learner.scenario_name AND vbs_sbs.fold = base_learner.fold AND vbs_sbs.metric = base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach='satzilla-11' GROUP BY scenario_name")

    print(satzilla)

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(10, 5))

    ax = fig.add_subplot(111)

    width = 0.3  # the width of the bars
    ind = np.arange(13)
    ax.bar(ind - width / 2, satzilla.result, width, color=color1, zorder=6)
    ax.bar(ind + width / 2, voting1234567.result, width, color=color3, zorder=6)

    plt.axhline(np.average(satzilla.result), color=color1, linestyle='dashed', linewidth=2, zorder=3)
    plt.axhline(np.average(voting1234567.result), color=color3, linestyle='dashed', linewidth=2, zorder=3)

    ax.set_xticks(ind)
    ax.set_xticklabels(["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013", "GLUHACK-18", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16\_INDU", "SAT12-INDU", "SAT18-EXP"])


    plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('nPAR10', fontsize=11)

    #ax.set_ylim(bottom=0.3)
    #ax.set_ylim(top=0.7)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    l1 = mpatches.Patch(color=color1, label="SATzilla'11")
    l2 = mpatches.Patch(color=color3, label="Voting")

    fig.legend(handles=[l1, l2], loc=1, prop={'size': 13}, bbox_to_anchor=(0.99, 0.98))
    plt.legend()
    plt.show()

    fig.savefig("plotted/voting_vs_baseline.pdf", bbox_inches='tight')


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