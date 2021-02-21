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

    #TODO: correct version for stacking normal??
    stacking1234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_4_5_6_7SUNNY' GROUP BY approach")
    stacking234567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_3_4_5_6_7SUNNY' GROUP BY approach")
    stacking134567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_3_4_5_6_7SUNNY' GROUP BY approach")
    stacking124567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_4_5_6_7SUNNY' GROUP BY approach")
    stacking123567 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_5_6_7SUNNY' GROUP BY approach")
    stacking123467 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_4_6_7SUNNY' GROUP BY approach")
    stacking123457 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_4_5_7SUNNY' GROUP BY approach")
    stacking123456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_4_5_6SUNNY' GROUP BY approach")

    fig = plt.figure(1, figsize=(24, 6))

    ax = fig.add_subplot(141)

    stacking_normal = float(stacking1234567.unsolved_instances) * 100

    plt.axhspan(stacking_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, stacking_normal, facecolor='g', alpha=0.2)

    plt.axhline(stacking_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking234567.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, stacking134567.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, stacking124567.unsolved_instances * 100, width, color=color1, label='ISAC')
    ax.bar(4, stacking123567.unsolved_instances * 100, width, color=color1, label='SATzilla-11')
    ax.bar(5, stacking123467.unsolved_instances * 100, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(6, stacking123457.unsolved_instances * 100, width, color=color1, label='SurvivalForestPAR10')
    ax.bar(7, stacking123456.unsolved_instances * 100, width, color=color1, label='Multiclass')

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla'11", "SF-Exp.", "SF-PAR10", "Multiclass"])

    ax.text(1, float(stacking234567.unsolved_instances) * 100, round(float(stacking234567.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(2, float(stacking134567.unsolved_instances) * 100, round(float(stacking134567.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(3, float(stacking124567.unsolved_instances) * 100 + 0.02, round(float(stacking124567.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(4, float(stacking123567.unsolved_instances) * 100, round(float(stacking123567.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(5, float(stacking123467.unsolved_instances) * 100, round(float(stacking123467.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(6, float(stacking123457.unsolved_instances) * 100, round(float(stacking123457.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)
    ax.text(7, float(stacking123456.unsolved_instances) * 100, round(float(stacking123456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom', rotation=0)

    #plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=6)
    ax.set_ylim(top=7)

    plt.title("Stacking with all base learners")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    stacking23456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_3_4_5_6SUNNY' GROUP BY approach")
    stacking13456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_3_4_5_6SUNNY' GROUP BY approach")
    stacking12456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_4_5_6SUNNY' GROUP BY approach")
    stacking12356 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_5_6SUNNY' GROUP BY approach")
    stacking12346 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_4_6SUNNY' GROUP BY approach")
    stacking12345 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_3_4_5SUNNY' GROUP BY approach")

    ax = fig.add_subplot(142)

    stacking_normal = float(stacking123456.unsolved_instances) * 100

    plt.axhspan(stacking_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, stacking_normal, facecolor='g', alpha=0.2)

    plt.axhline(stacking_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking23456.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, stacking13456.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, stacking12456.unsolved_instances * 100, width, color=color1, label='ISAC')
    ax.bar(4, stacking12356.unsolved_instances * 100, width, color=color1, label='SATzilla-11')
    ax.bar(5, stacking12346.unsolved_instances * 100, width, color=color1, label='SurvivalForestExpectation')
    ax.bar(6, stacking12345.unsolved_instances * 100, width, color=color1, label='SurvivalForestPAR10')

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla", "SF-Exp.", "SF-PAR10"])

    ax.text(1, float(stacking23456.unsolved_instances) * 100,
            round(float(stacking23456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(2, float(stacking13456.unsolved_instances) * 100,
            round(float(stacking13456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(3, float(stacking12456.unsolved_instances) * 100,
            round(float(stacking12456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(4, float(stacking12356.unsolved_instances) * 100,
            round(float(stacking12356.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(5, float(stacking12346.unsolved_instances) * 100,
            round(float(stacking12346.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(6, float(stacking12345.unsolved_instances) * 100,
            round(float(stacking12345.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)

    # plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=6)
    ax.set_ylim(top=7)

    plt.title("Stacking without base learner 'Multiclass'")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    stacking2456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_4_5_6SUNNY' GROUP BY approach")
    stacking1456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_4_5_6SUNNY' GROUP BY approach")
    stacking1256 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_5_6SUNNY' GROUP BY approach")
    stacking1246 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_4_6SUNNY' GROUP BY approach")
    stacking1245 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_1_2_4_5SUNNY' GROUP BY approach")

    ax = fig.add_subplot(143)

    stacking_normal = float(stacking12456.unsolved_instances) * 100

    plt.axhspan(stacking_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, stacking_normal, facecolor='g', alpha=0.2)

    plt.axhline(stacking_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking2456.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, stacking1456.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, stacking1256.unsolved_instances * 100, width, color=color1, label='ISAC')
    ax.bar(4, stacking1246.unsolved_instances * 100, width, color=color1, label='SATzilla-11')
    ax.bar(5, stacking1245.unsolved_instances * 100, width, color=color1, label='SurvivalForestExpectation')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["PerAlgo", "SUNNY", "SATzilla", "SF-Exp.", "SF-PAR10"])

    ax.text(1, float(stacking2456.unsolved_instances) * 100,
            round(float(stacking2456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(2, float(stacking1456.unsolved_instances) * 100,
            round(float(stacking1456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(3, float(stacking1256.unsolved_instances) * 100,
            round(float(stacking1256.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(4, float(stacking1246.unsolved_instances) * 100,
            round(float(stacking1246.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(5, float(stacking1245.unsolved_instances) * 100 ,
            round(float(stacking1245.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)

    # plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=6)
    ax.set_ylim(top=7)

    plt.title("Stacking without base learners 'Multiclass' and 'ISAC'")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    stacking456 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_4_5_6SUNNY' GROUP BY approach")
    stacking256 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_5_6SUNNY' GROUP BY approach")
    stacking246 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_4_6SUNNY' GROUP BY approach")
    stacking245 = get_dataframe_for_sql_query(
        "SELECT approach, AVG(result) as unsolved_instances FROM stacking WHERE metric='number_unsolved_instances_False' AND approach='stacking_2_4_5SUNNY' GROUP BY approach")

    ax = fig.add_subplot(144)

    stacking_normal = float(stacking2456.unsolved_instances) * 100

    plt.axhspan(stacking_normal, 100, facecolor='r', alpha=0.2)
    plt.axhspan(0, stacking_normal, facecolor='g', alpha=0.2)

    plt.axhline(stacking_normal, color='#000', linestyle='dashed', linewidth=2)

    width = 0.4  # the width of the bars
    ax.bar(1, stacking456.unsolved_instances * 100, width, color=color1, label='PerAlgorithmRegressor')
    ax.bar(2, stacking256.unsolved_instances * 100, width, color=color1, label='SUNNY')
    ax.bar(3, stacking246.unsolved_instances * 100, width, color=color1, label='ISAC')
    ax.bar(4, stacking245.unsolved_instances * 100, width, color=color1, label='SATzilla-11')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["SUNNY", "SATzilla", "SF-Exp.", "SF-PAR10"])

    ax.text(1, float(stacking456.unsolved_instances) * 100,
            round(float(stacking456.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(2, float(stacking256.unsolved_instances) * 100,
            round(float(stacking256.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(3, float(stacking246.unsolved_instances) * 100,
            round(float(stacking246.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)
    ax.text(4, float(stacking245.unsolved_instances) * 100,
            round(float(stacking245.unsolved_instances) * 100 - stacking_normal, 3), ha='center', va='bottom',
            rotation=0)

    # plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=6)
    ax.set_ylim(top=7)

    plt.title("Stacking without base learners 'Multiclass', 'ISAC' and 'PerAlgo'")
    plt.xlabel("left out base learner")
    plt.ylabel("unsolved instances (%)")

    plt.show()

    fig.savefig("plotted/stacking_base_learner_usi.pdf", bbox_inches='tight')


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