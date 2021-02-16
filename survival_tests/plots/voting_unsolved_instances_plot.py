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
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_3_4_5_6_7' GROUP BY approach")
    voting_weighting_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_weighting_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_weighting_1_2_3_4_5_6_7' GROUP BY approach")
    voting_cross_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_cross WHERE metric='number_unsolved_instances_False' AND approach='voting_weighting_cross_1_2_3_4_5_6_7' GROUP BY approach")
    voting_ranking_1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_ranking WHERE metric='number_unsolved_instances_False' AND approach='voting_ranking_1_2_3_4_5_6_7' GROUP BY approach")

    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_2_4_5_6_7' GROUP BY approach")
    voting_weighting_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_weighting_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_weighting_2_4_5_6_7' GROUP BY approach")
    voting_cross_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_cross WHERE metric='number_unsolved_instances_False' AND approach='voting_weighting_cross_2_4_5_6_7' GROUP BY approach")
    voting_ranking_24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_ranking WHERE metric='number_unsolved_instances_False' AND approach='voting_ranking_2_4_5_6_7' GROUP BY approach")

    plt.rc('font', family='sans-serif')
    plt.rc('text', usetex=True)

    fig = plt.figure(1, figsize=(7, 5))

    ax = fig.add_subplot(111)

    multiplyer = 100

    width = 0.18  # the width of the bars
    ax.bar(0.7, multiplyer * voting1234567.unsolved_instances, width, color=color1, zorder=6)
    ax.bar(0.9, multiplyer * voting_weighting_1234567.unsolved_instances, width, color=color2, zorder=6)
    ax.bar(1.1, multiplyer * voting_cross_1234567.unsolved_instances, width, color=color3, zorder=6)
    ax.bar(1.3, multiplyer * voting_ranking_1234567.unsolved_instances, width, color=color4, zorder=6)
    ax.bar(1.7, multiplyer * voting24567.unsolved_instances, width, color=color1, zorder=6)
    ax.bar(1.9, multiplyer * voting_weighting_24567.unsolved_instances, width, color=color2, zorder=6)
    ax.bar(2.1, multiplyer * voting_cross_24567.unsolved_instances, width, color=color3, zorder=6)
    ax.bar(2.3, multiplyer * voting_ranking_24567.unsolved_instances, width, color=color4, zorder=6)

    ax.text(0.7, float(multiplyer * voting1234567.unsolved_instances),
            round(float(multiplyer * voting1234567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(0.9, float(multiplyer * voting_weighting_1234567.unsolved_instances),
            round(float(multiplyer * voting_weighting_1234567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(1.1, float(multiplyer * voting_cross_1234567.unsolved_instances),
            round(float(multiplyer * voting_cross_1234567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(1.3, float(multiplyer * voting_ranking_1234567.unsolved_instances),
            round(float(multiplyer * voting_ranking_1234567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(1.7, float(multiplyer * voting24567.unsolved_instances),
            round(float(multiplyer * voting24567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(1.9, float(multiplyer * voting_weighting_24567.unsolved_instances),
            round(float(multiplyer * voting_weighting_24567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(2.1, float(multiplyer * voting_cross_24567.unsolved_instances),
            round(float(multiplyer * voting_cross_24567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)
    ax.text(2.3, float(multiplyer * voting_ranking_24567.unsolved_instances),
            round(float(multiplyer * voting_ranking_24567.unsolved_instances), 3), ha='center',
            va='bottom', rotation=0)


    ax.set_xticks([1,2])
    ax.set_xticklabels(["Voting", "Selected Voting"])


    #plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('number of unsolved instances', fontsize=11)

    ax.set_ylim(bottom=4)
    ax.set_ylim(top=7)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    l1 = mpatches.Patch(color=color1, label="Majority Voting")
    l2 = mpatches.Patch(color=color2, label="Weighted Voting")
    l3 = mpatches.Patch(color=color3, label="Weighted Voting (cross-validation)")
    l4 = mpatches.Patch(color=color4, label="Ranked Voting")

    plt.legend(handles=[l1, l2, l3, l4], loc=2)
    plt.show()

    fig.savefig("plotted/voting_unsolved_instances.pdf", bbox_inches='tight')


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