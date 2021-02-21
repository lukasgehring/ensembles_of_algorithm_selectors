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
    voting1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_3_4_5_6_7' GROUP BY approach")
    voting234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_2_3_4_5_6_7' GROUP BY approach")
    voting134567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_3_4_5_6_7' GROUP BY approach")
    voting124567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_4_5_6_7' GROUP BY approach")
    voting123567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_3_5_6_7' GROUP BY approach")
    voting123467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_3_4_6_7' GROUP BY approach")
    voting123457 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_3_4_5_7' GROUP BY approach")
    voting123456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_3_4_5_6' GROUP BY approach")

    fig = plt.figure(1, figsize=(18, 6))

    ax = fig.add_subplot(131)

    voting_normal = float(voting1234567.unsolved_instances) * 100

    plt.axhspan(voting_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, voting234567.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, voting134567.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, voting124567.unsolved_instances * 100, width, color=color1, label='ISAC')
    ax.bar(4, voting123567.unsolved_instances * 100, width, color=color1, label='SATzilla-11')
    ax.bar(5, voting123467.unsolved_instances * 100, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(6, voting123457.unsolved_instances * 100, width, color=color1, label='SurvivalForestPAR10')
    ax.bar(7, voting123456.unsolved_instances * 100, width, color=color1, label='Multiclass')

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla'11", "SF-Exp.", "SF-PAR10", "Multiclass"])

    ax.text(1, float(voting234567.unsolved_instances) * 100, round(float(voting234567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(voting134567.unsolved_instances) * 100, round(float(voting134567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(voting124567.unsolved_instances) * 100, round(float(voting124567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(voting123567.unsolved_instances) * 100, round(float(voting123567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(5, float(voting123467.unsolved_instances) * 100 + 0.02, round(float(voting123467.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(6, float(voting123457.unsolved_instances) * 100, round(float(voting123457.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(7, float(voting123456.unsolved_instances) * 100, round(float(voting123456.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom', rotation=0)

    #plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=5)
    ax.set_ylim(top=6.5)

    plt.title("Voting with all base learners")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    voting24567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_2_4_5_6_7' GROUP BY approach")
    voting14567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_4_5_6_7' GROUP BY approach")
    voting12567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_5_6_7' GROUP BY approach")
    voting12467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_4_6_7' GROUP BY approach")
    voting12457 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_4_5_7' GROUP BY approach")
    voting12456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_4_5_6' GROUP BY approach")

    ax = fig.add_subplot(132)

    voting_normal = float(voting124567.unsolved_instances) * 100

    plt.axhspan(voting_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, voting24567.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, voting14567.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, voting12567.unsolved_instances * 100, width, color=color1, label='SATzilla-11')
    ax.bar(4, voting12467.unsolved_instances * 100, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(5, voting12457.unsolved_instances * 100, width, color=color1, label='SurvivalForestPAR10')
    ax.bar(6, voting12456.unsolved_instances * 100, width, color=color1, label='Multiclass')

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "SATzilla", "SF-Exp.", "SF-PAR10", "Multiclass"])

    ax.text(1, float(voting24567.unsolved_instances) * 100,
            round(float(voting24567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(2, float(voting14567.unsolved_instances) * 100,
            round(float(voting14567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(3, float(voting12567.unsolved_instances) * 100,
            round(float(voting12567.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(4, float(voting12467.unsolved_instances) * 100 + 0.04,
            round(float(voting12467.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(5, float(voting12457.unsolved_instances) * 100,
            round(float(voting12457.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(6, float(voting12456.unsolved_instances) * 100,
            round(float(voting12456.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)

    # plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=5)
    ax.set_ylim(top=6.5)

    plt.title("Voting without base learner 'ISAC'")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    voting2467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_2_4_6_7' GROUP BY approach")
    voting1467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_4_6_7' GROUP BY approach")
    voting1267 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_6_7' GROUP BY approach")
    voting1247 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_4_7' GROUP BY approach")
    voting1246 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM voting_base_learner_selection WHERE metric='number_unsolved_instances_False' AND approach='voting_1_2_4_6' GROUP BY approach")

    ax = fig.add_subplot(133)

    voting_normal = float(voting12467.unsolved_instances) * 100

    plt.axhspan(voting_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, voting_normal, facecolor='g', alpha=0.2)

    plt.axhline(voting_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, voting2467.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, voting1467.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, voting1267.unsolved_instances * 100, width, color=color1, label='SATzilla-11')
    ax.bar(4, voting1247.unsolved_instances * 100, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(5, voting1246.unsolved_instances * 100, width, color=color1, label='SurvivalForestPAR10')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "SATzilla", "SF-PAR10", "Multiclass"])

    ax.text(1, float(voting2467.unsolved_instances) * 100,
            round(float(voting2467.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(2, float(voting1467.unsolved_instances) * 100,
            round(float(voting1467.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(3, float(voting1267.unsolved_instances) * 100,
            round(float(voting1267.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(4, float(voting1247.unsolved_instances) * 100,
            round(float(voting1247.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(5, float(voting1246.unsolved_instances) * 100,
            round(float(voting1246.unsolved_instances) * 100 - voting_normal, 3), ha='center', va='bottom',
            rotation=0)

    # plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=5)
    ax.set_ylim(top=6.5)

    plt.title("Voting without base learners 'ISAC' and 'SF-Exp.'")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    plt.show()

    fig.savefig("plotted/voting_base_learner_usi.pdf", bbox_inches='tight')


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