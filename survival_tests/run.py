import logging
import sys
import configparser
import multiprocessing as mp

from sklearn.ensemble import RandomForestRegressor

import database_utils
from ensembles.bagging import Bagging
from ensembles.boosting import Boosting
from ensembles.create_base_learner import CreateBaseLearner
from ensembles.stacking import Stacking
from ensembles.voting import Voting
from ensembles.voting_cross_validation import Voting_Cross
from ensembles.voting_pre_computed import VotingPreComputed
from evaluation import evaluate_scenario
from approaches.single_best_solver import SingleBestSolver
from approaches.oracle import Oracle
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.sunny import SUNNY
from baselines.snnap import SNNAP
from baselines.isac import ISAC
from baselines.satzilla11 import SATzilla11
from baselines.satzilla07 import SATzilla07
from sklearn.linear_model import Ridge
from par_10_metric import Par10Metric
from number_unsolved_instances import NumberUnsolvedInstances


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiements for scenario: " + result)


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        if approach_name == 'sbs':
            approaches.append(SingleBestSolver())
        if approach_name == 'oracle':
            approaches.append(Oracle())
        if approach_name == 'ExpectationSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Expectation'))
        if approach_name == 'PolynomialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Polynomial'))
        if approach_name == 'GridSearchSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='GridSearch'))
        if approach_name == 'ExponentialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Exponential'))
        if approach_name == 'SurrogateAutoSurvivalForest':
            approaches.append(SurrogateAutoSurvivalForest())
        if approach_name == 'PAR10SurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='PAR10'))
        if approach_name == 'per_algorithm_regressor':
            approaches.append(PerAlgorithmRegressor())
        if approach_name == 'imputed_per_algorithm_rf_regressor':
            approaches.append(PerAlgorithmRegressor(impute_censored=True))
        if approach_name == 'imputed_per_algorithm_ridge_regressor':
            approaches.append(PerAlgorithmRegressor(
                scikit_regressor=Ridge(alpha=1.0), impute_censored=True))
        if approach_name == 'multiclass_algorithm_selector':
            approaches.append(MultiClassAlgorithmSelector())
        if approach_name == 'sunny':
            approaches.append(SUNNY())
        if approach_name == 'snnap':
            approaches.append(SNNAP())
        if approach_name == 'satzilla-11':
            approaches.append(SATzilla11())
        if approach_name == 'satzilla-07':
            approaches.append(SATzilla07())
        if approach_name == 'isac':
            approaches.append(ISAC())
        if approach_name == 'voting':
            approaches.append(Voting())
        if approach_name == 'voting_rank':
            approaches.append(Voting(ranking=True))
        if approach_name == 'voting_rank_min':
            approaches.append(Voting(ranking=True, rank_method='min'))
        if approach_name == 'voting_weight':
            approaches.append(Voting(weighting=True))
        if approach_name == 'voting_weight_cross':
            approaches.append(Voting(weighting=True, cross_validation=True))
        if approach_name == 'voting_base_learner_test':
            approaches.append(Voting(base_learner_test=1))
            approaches.append(Voting(base_learner_test=2))
            approaches.append(Voting(base_learner_test=3))
            approaches.append(Voting(base_learner_test=4))
            approaches.append(Voting(base_learner_test=5))
            approaches.append(Voting(base_learner_test=6))
        if approach_name == 'bagging-per_algorithm_regressor':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor()))
        if approach_name == 'bagging-per_multiclass_algorithm_selector':
            approaches.append(Bagging(num_base_learner=10, base_learner=MultiClassAlgorithmSelector()))
        if approach_name == 'bagging-per_algorithm_regressor_weight':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor(), weighting=True))
        if approach_name == 'bagging-SUNNY_weight':
            approaches.append(Bagging(num_base_learner=10, base_learner=SUNNY(), weighting=True))
        if approach_name == 'bagging-decision_tree':
            approaches.append(Bagging(num_base_learner=200, base_learner=PerAlgorithmRegressor(scikit_regressor=RandomForestRegressor(n_jobs=1, n_estimators=1, bootstrap=False))))
            approaches.append(Bagging(num_base_learner=400, base_learner=PerAlgorithmRegressor(
                scikit_regressor=RandomForestRegressor(n_jobs=1, n_estimators=1, bootstrap=False))))
        if approach_name == 'bagging-per_algorithm_regressor_rank':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor(), use_ranking=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor(), weighting=True, use_ranking=True))
        if approach_name == 'bagging-per_algorithm_regressor_performance_rank':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor(), use_ranking=True, performance_ranking=True))
        if approach_name == 'bagging-satzilla-11':
            approaches.append(Bagging(num_base_learner=10, base_learner=SATzilla11()))
        if approach_name == 'bagging-ExpectationSurvivalForest':
            approaches.append(Bagging(num_base_learner=10, base_learner=SurrogateSurvivalForest(criterion='Expectation')))
        if approach_name == 'bagging-SUNNY':
            approaches.append(Bagging(num_base_learner=10, base_learner=SUNNY()))
        if approach_name == 'bagging-ISAC':
            approaches.append(Bagging(num_base_learner=10, base_learner=ISAC()))
        if approach_name == 'bagging-number_of_base_learner':
            approaches.append(Bagging(num_base_learner=4, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=8, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=12, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=16, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=20, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=24, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=28, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=32, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=36, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=40, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=44, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=48, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=52, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=56, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=60, base_learner=PerAlgorithmRegressor()))
        if approach_name == 'bagging-number_of_base_learner_SUNNY':
            approaches.append(Bagging(num_base_learner=4, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=8, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=12, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=16, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=20, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=24, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=28, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=32, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=36, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=40, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=44, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=48, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=52, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=56, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=60, base_learner=SUNNY()))
        if approach_name == 'bagging-number_of_base_learner_SATzilla':
            approaches.append(Bagging(num_base_learner=4, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=8, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=12, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=16, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=20, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=24, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=28, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=32, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=36, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=40, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=44, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=48, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=52, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=56, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=60, base_learner=SATzilla11()))
        if approach_name == 'boosting':
            approaches.append(Boosting('per_algorithm_regressor'))
        if approach_name == 'adaboost_stumpt':
            approaches.append(Boosting('per_algorithm_regressor', stump=True))
        if approach_name == 'adaboost_stump_100':
            approaches.append(Boosting('per_algorithm_regressor', num_iterations=100, stump=True))
        if approach_name == 'adaboost_stump_500':
            approaches.append(Boosting('per_algorithm_regressor', num_iterations=500, stump=True))
        if approach_name == 'boosting_stumpt':
            approaches.append(Boosting('per_algorithm_regressor', stump=True, singlelearner=True))
        if approach_name == 'boosting_stump_100':
            approaches.append(Boosting('per_algorithm_regressor', num_iterations=100, stump=True, singlelearner=True))
        if approach_name == 'boosting_multiclass':
            approaches.append(Boosting('multiclass_algorithm_selector'))
        if approach_name == 'boosting_multiclass_100':
            approaches.append(Boosting('multiclass_algorithm_selector', num_iterations=100))
        if approach_name == 'boosting_ExponentialSurvivalForest':
            approaches.append(Boosting('ExponentialSurvivalForest'))
        if approach_name == 'stacking':
            approaches.append(Stacking())
        if approach_name == 'stacking_with_VarianceThreshold':
            approaches.append(Stacking(feature_selection='VarianceThreshold'))
        if approach_name == 'stacking_with_SelectKBest_f_regression':
            approaches.append(Stacking(feature_selection='SelectKBest_f_regression'))
        if approach_name == 'stacking_with_SelectKBest_mutual_info_regression':
            approaches.append(Stacking(feature_selection='SelectKBest_mutual_info_regression'))
        if approach_name == 'stacking_cross_validation':
            approaches.append(Stacking(cross_validation=True))
        if approach_name == 'stacking_multiclass':
            approaches.append(Stacking(meta_learner_type='multiclass_algorithm_selector'))

        if approach_name == 'create_base_learner':
            approaches.append(CreateBaseLearner(algorithm='per_algorithm_regressor'))
            approaches.append(CreateBaseLearner(algorithm='sunny'))
            approaches.append(CreateBaseLearner(algorithm='isac'))
            approaches.append(CreateBaseLearner(algorithm='satzilla'))
            approaches.append(CreateBaseLearner(algorithm='expectation'))
            approaches.append(CreateBaseLearner(algorithm='par10'))
            approaches.append(CreateBaseLearner(algorithm='multiclass'))
        if approach_name == 'voting_pre_computed':
            approaches.append(VotingPreComputed())
    return approaches


#######################
#         MAIN        #
#######################

initialize_logging()
config = load_configuration()
logger.info("Running experiments with config:")
print_config(config)

#fold = int(sys.argv[1])
#logger.info("Running experiments for fold " + str(fold))

db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(
    config)
database_utils.create_table_if_not_exists(db_handle, table_name)

amount_of_cpus_to_use = int(config['EXPERIMENTS']['amount_of_cpus'])
pool = mp.Pool(amount_of_cpus_to_use)


scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
approach_names = config["EXPERIMENTS"]["approaches"].split(",")
amount_of_scenario_training_instances = int(
    config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
tune_hyperparameters = bool(int(config["EXPERIMENTS"]["tune_hyperparameters"]))

for fold in range(1, 11):

    for scenario in scenarios:
        approaches = create_approach(approach_names)

        if len(approaches) < 1:
            logger.error("No approaches recognized!")
        for approach in approaches:
            metrics = list()
            metrics.append(Par10Metric())
            if approach.get_name() != 'oracle':
                metrics.append(NumberUnsolvedInstances(False))
                metrics.append(NumberUnsolvedInstances(True))
            logger.info("Submitted pool task for approach \"" +
                        str(approach.get_name()) + "\" on scenario: " + scenario)
            pool.apply_async(evaluate_scenario, args=(scenario, approach, metrics,
                                                      amount_of_scenario_training_instances, fold, config, tune_hyperparameters), callback=log_result)

            #evaluate_scenario(scenario, approach, metrics,
            #                 amount_of_scenario_training_instances, fold, config, tune_hyperparameters)
            print('Finished evaluation of fold')

pool.close()
pool.join()
logger.info("Finished all experiments.")
