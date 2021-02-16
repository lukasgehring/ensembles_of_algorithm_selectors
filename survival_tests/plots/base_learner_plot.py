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

    nPAR10 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, pre_computed_base_learner.approach, vbs_sbs.metric, pre_computed_base_learner.result, ((pre_computed_base_learner.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN pre_computed_base_learner ON vbs_sbs.scenario_name = pre_computed_base_learner.scenario_name AND vbs_sbs.fold = pre_computed_base_learner.fold AND vbs_sbs.metric = pre_computed_base_learner.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    unsolved_instances = get_dataframe_for_sql_query(
        "SELECT s.approach, AVG(s.unsolved_instances) as result FROM (SELECT scenario_name, AVG(result) as unsolved_instances, approach FROM `pre_computed_base_learner` WHERE metric='number_unsolved_instances_False' GROUP BY scenario_name, approach) as s GROUP BY s.approach")

    print(nPAR10)
    print(unsolved_instances)
    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(7, 5))

    ax1 = fig.add_subplot(111)

    names = list()
    names.append("SF-Exp")
    names.append("ISAC")
    names.append("Multiclass")
    names.append("SF-PAR10")
    names.append("PerAlgo")
    names.append("SATzilla'11")
    names.append("SUNNY")

    width = 0.5  # the width of the bars
    ind = [4, 2, 6, 5, 0, 3, 1]
    ax1.bar(ind, nPAR10.result, width, color=color1, label='Single Learner', zorder=6)
    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    ax1.set_xticks(ind)
    ax1.set_xticklabels(names)
    ax1.tick_params(axis='y', colors=color1)

    for i, value in zip(ind, nPAR10.result):
        ax1.text(i, value, round(value, 2), ha='center', va='bottom', rotation=0)


    plt.xlabel('Learing Algorithm')
    plt.ylabel('nPAR10')

    ax1.set_ylim(bottom=0)
    ax1.set_ylim(top=1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('unsolved instances (\%)')
    ax2.plot(ind, unsolved_instances.result*100, marker='s', lw=0, color=color3)
    ax2.set_yticks(np.arange(6)*4)
    ax2.tick_params(axis='y', colors=color3)

    plt.show()

    fig.savefig("plotted/baselines.pdf", bbox_inches='tight')


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