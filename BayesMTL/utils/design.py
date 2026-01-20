#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:00:05 2018

@author: goncalves1
"""
import os
import types
import pickle
import shutil
from collections import defaultdict
from abc import ABCMeta, abstractmethod

import ray
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import precision_recall_curve, confusion_matrix

from bayesmtl.utils import performance_metrics
from bayesmtl.utils.Logger import Logger
from bayesmtl.utils.hyperparameter_selection import select_best_hyperparameters
import bayesmtl.utils.constants as const

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'figure.max_open_warning': 0})

# metrics improvement sense
THE_HIGHER_THE_BETTER = {'area_under_curve', 'avg_precision', 'weighted_accuracy', 'accuracy', 'accuracy_per_class', 'c_index', 'c_index_ours'}
THE_LOWER_THE_BETTER = {'rmse', 'nmse', 'rmse_survival', 'mse_survival', 'brier_score', 'mae_survival'}

# maximum and minimum relative performance improvement
# it is necessary to avoid large differences that distorts the plots
MAX_REL_IMPROVEMENT = 250

class ModelTraining(object):
    """
    """
    def __init__(self, name, dataset_name="simulated"):

        assert isinstance(name, str)

        self.name = name
        self.dataset_name = dataset_name
        self.dataset = None
        self.methods = None
        self.metrics = None
        self.nb_runs = -1
        self.nb_tasks = -1

        self.logger = Logger()
        
    def execute(self, training_config, dataset, models_classes,
            models_params, method_names, only_report=False, num_cpus=4,
            resume_training=False, data_subtype=None, data_class=None, data_path=None,
            training=True, calibration=False, data_version=None):
        self.training_config = training_config
        self.dataset = dataset
        load_params = False
        self.methods = [m(name=name) for m,name in zip(models_classes,method_names)]
        self.metrics = self.training_config['metrics']
        self.nb_runs = self.training_config['number_runs']
        self.__check_inputs(dataset, models_classes, self.metrics, self.nb_runs)

        if only_report:
            return 0
        # set experiment output directory
        directory = os.path.join(os.getcwd(), "log", self.name)
        directory = os.path.join(directory, data_subtype)

        if not resume_training: # new training folder

            # if directory already exists, then delete it
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)

            # make a new directory with experiment name
            os.makedirs(directory)

            # experiment log file will be save in 'directory'
            self.logger.set_path(directory)
            self.logger.setup_logger('{}.log'.format(self.name))
            self.logger.info('Experiment directory created.')

        else:
            
            os.makedirs(directory,exist_ok=True)
            self.logger.set_path(directory)
            self.logger.setup_logger('{}_resume.log'.format(self.name))
            self.logger.info('Resume training is ON. Not recreating the folder ...')

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}
        ray.init(num_cpus=num_cpus)
        for r_i in tqdm(range(self.nb_runs)): 
            load_params = False
            self.dataset = data_class(data_path,
                    data_subtype = data_subtype,
                    r_i=r_i)
            self.nb_tasks = len(self.dataset.data_split['train']['y'])
            self.logger.info('Executing \'Run {}\'.'.format(self.dataset.r_i))
            run_directory = os.path.join(directory, 'run_{}'.format(self.dataset.r_i))

            # execute all methods passed through 'methods' attribute
            for m_idx, method in enumerate(self.methods):

                # set method's output directory
                method_directory = os.path.join(run_directory, method.__str__())

                print("The method directory is {}".format(method_directory), flush=True)
                print("The method name is {}".format(method.name), flush=True)

                if resume_training:
                    # has the method in this run been completed? if so skip it otherwise run it
                    r_code = self.check_method_completed(method_directory)
                    print("The method complete status is {}".format(r_code))
                    if r_code == 0:
                        # skip to the next run
                        if method.paradigm=="MTL":
                            # For MTL method check if the model are already trained with
                            # sufficient iterations
                            max_iters = np.load(method_directory+'/max_iter.npy')
                            if max_iters>=method.max_iters:
                                self.logger.info("Run {} for method {} already completed. Moving to the next ...".format(r_i+1,method.name))
                                # training = False
                                continue
                        else:
                            self.logger.info("Run {} for method {} already completed. Moving to the next ...".format(r_i+1,method.name))
                            continue
                    elif r_code ==-2:
                        # check if training is completed if so move to
                        # prediction only
                        if method.paradigm=="MTL":
                            # For MTL method check if the model are already trained with
                            # sufficient iterations
                            max_iters = np.load(method_directory+'/max_iter.npy')
                            if max_iters>=method.max_iters:
                                self.logger.info("Training {} for method {} completed . Moving to the prediction ...".format(r_i+1,method.name))
                                training = False
                        else:
                            self.logger.info("Training {} for method {} completed . Moving to the prediction ...".format(r_i+1,method.name))
                            training = False

                    else:
                        self.logger.info("Run {} for method {} partially or not completed. Running it [Code {}].".format(r_i+1, method.name, r_code))

                        # delete folder and run experiment from scratch if hyperparameter files is not found
                        if os.path.exists(method_directory):
                            if os.path.exists(os.path.join(method_directory,'{}_hp_tuning.csv'.format(method.name))):
                                for filename in os.listdir(method_directory):
                                    if filename.endswith(".csv"):
                                        continue  # Skip .csv files, do not remove them
                                    file_path = os.path.join(method_directory, filename)
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                            else:
                                shutil.rmtree(method_directory,ignore_errors=True)

                self.logger.info('Running {}.'.format(method.name))

                if method.paradigm=="MTL":
                    method.init_with_data(self.dataset.data_split['train']['x'],self.dataset.data_split['train']['y'],intercept=False,seed=r_i)
                    if not training:
                        method.set_mode('train')
                        method.test=True

                    # create directory if not existent already to save method's results/logs
                    # otherwise load the previously trained parameters
                    if os.path.exists(method_directory+'/{}.mdl'.format(method.name)):
                        print("Found existing directory {}".format(method_directory),flush=True)
                        print("start loading parameters...",flush=True)
                        params_name = const.params_dict[method.name]
                        with open(method_directory+'/{}.mdl'.format(method.name), 'rb') as f:
                            # The protocol version used is detected automatically, so we do not
                            # have to specify it.
                            model_trained = pickle.load(f)
                        params = dict(zip(params_name,model_trained))
                        method.load_params(params)
                        load_params = True
                        print("parameters loaded!",flush=True)
                    else:
                        if not os.path.isdir(method_directory):
                            print("Making directory {}".format(method_directory),flush=True)
                            os.makedirs(method_directory)
                else:
                    try:
                        print("Making directory {}".format(method_directory),flush=True)
                        os.makedirs(method_directory)
                    except FileExistsError:
                        pass
                # inform output directory path to the method
                method.set_output_directory(method_directory)
                sample_sizes = [self.dataset.data_split['train']['x'][t].shape[0]for t in range(self.nb_tasks)]

                # check model paradigm: STL or MTL
                if method.paradigm == 'STL':
                    # dict to store performance metrics for all tasks
                    # obtained using 'method'
                    results = {}
                    ypred_yobs = {}
                    feature_importance = {}
                    if not training:
                        fname = os.path.join(method_directory,
                            '{}.mdl'.format(method.__str__()))
                        with open(fname, 'rb') as f:
                            model_trained = pickle.load(f)
                    if method.name=="STL-LC":
                        weights = np.empty((self.nb_tasks,self.dataset.data_split['train']['x'][0].shape[1]+1))
                    for t in range(self.nb_tasks):
                        if training:
                            if self.training_config['hyperparameter_selection']['on']:
                                method.set_mode('debug')
                                num_trials = self.training_config['hyperparameter_selection']['num_trials']
                                mode_hp = self.training_config['hyperparameter_selection']['mode']
                                metric_hp = metric_func[self.training_config['hyperparameter_selection']['metric']]
                                k_hp = self.training_config['hyperparameter_selection']['k']
                                num_cpu_hp = self.training_config['hyperparameter_selection']['num_cpu']
                                best_hyper_params = select_best_hyperparameters(X=self.dataset.data_split['train']['x'][t],
                                                                                Y=self.dataset.data_split['train']['y'][t],
                                                                                model_class=models_classes[m_idx],
                                                                                params=models_params[m_idx],
                                                                                metric=metric_hp,
                                                                                mode=mode_hp,
                                                                                # num_samples=num_trials//30,
                                                                                num_samples=int(num_trials**(1/3)),
                                                                                base_model = method,
                                                                                output_dir=method.output_directory,
                                                                                name = "{}-{}-{}-{}".format(self.name,method.name,r_i,data_subtype),
                                                                                run=r_i, k_hp=k_hp,
                                                                                num_cpu = num_cpu_hp,
                                                                                t=t
                                                                                )
                                # set to best hyperparameters found
                                method.set_params(best_hyper_params)
                            else:
                                # if not doing hyperparameter search, take the first element in the list
                                hp_space = method.get_hyperparameter_space()
                                method.set_params({k:v[0] for k,v in hp_space['space'].items()})
                            method.set_mode('train')
                            method.fit(self.dataset.data_split['train']['x'][t],
                                self.dataset.data_split['train']['y'][t],
                                intercept=False,
                                column_names=None,
                                run=r_i, task_id=t)
                            method.load_params(model_trained[t,:])
                            method.model.classes_ = np.unique(self.dataset.data_split['train']['y'][t])
                            method.column_names=self.dataset.column_names
                            # store feature importance to output folder
                            feat_imp = method.feature_importance()
                            feature_importance[self.dataset.data_split['task_name'][t]] = feat_imp
                        y_pred = method.predict(self.dataset.data_split['test']['x'][t], task_id=t)
                        ypred_yobs[t] = {'pred': y_pred,
                                        'obs': self.dataset.data_split['test']['y'][t]}
                        result_task = {}
                        for met in self.metrics:
                            y_true = self.dataset.data_split['test']['y'][t]
                            result_task[met] = metric_func[met](y_pred,
                                                                y_true)
                                                                # censor_flag=self.dataset.data['test']['censor_flag'][t],
                                                                 # survival_time=self.dataset.data['test']['svv_time'][t])
                        n_samples = self.dataset.data_split['train']['x'][t].shape[0]
                        # save result
                        results[t] = {'results': result_task,
                                        'sample_size': n_samples}
                        if method.name=="STL-LC" and training:
                            weights[t,:] = method.return_weight()
                    if method.name=="STL-LC" and training:
                        fname = os.path.join(method_directory,
                            '{}.mdl'.format(method.__str__()))
                        with open(fname, 'wb') as fh:
                            pickle.dump(weights, fh)
                elif method.paradigm in ('MTL', 'Pooled'):
                    # check whether it's in training state or in
                    # prediction state
                    if training:
                        if method.paradigm!="MTL" or load_params==False :
                            if self.training_config['hyperparameter_selection']['on']:
                                method.set_mode('debug')
                                num_trials = self.training_config['hyperparameter_selection']['num_trials']
                                mode_hp = self.training_config['hyperparameter_selection']['mode']
                                metric_hp = metric_func[self.training_config['hyperparameter_selection']['metric']]
                                k_hp = self.training_config['hyperparameter_selection']['k']
                                num_cpu_hp = self.training_config['hyperparameter_selection']['num_cpu']
                                best_hyper_params = select_best_hyperparameters(X=self.dataset.data_split['train']['x'],
                                                                                Y=self.dataset.data_split['train']['y'],
                                                                                model_class=models_classes[m_idx],
                                                                                params=models_params[m_idx],
                                                                                metric=metric_hp,
                                                                                mode=mode_hp,
                                                                                num_samples=num_trials,
                                                                                # num_samples=num_trials if method.name=="BayesMTL-mod2" else num_trials//10,
                                                                                base_model = method,
                                                                                output_dir=method.output_directory,
                                                                                name = "{}-{}-{}-{}".format(self.name,method.name,r_i,data_subtype),
                                                                                run=r_i,
                                                                                k_hp = k_hp,
                                                                                num_cpu = num_cpu_hp
                                                                                )
                                # set to best hyperparameters found
                                method.set_params(best_hyper_params)
                            else:
                                # if not doing hyperparameter search, take the first element in the list
                                hp_space = method.get_hyperparameter_space()
                                method.set_params({k:v for k,v in hp_space.items()})
                                # method.set_params({k:v[0] for k,v in hp_space['space'].items()})
                        method.set_mode('train')
                        method.fit(self.dataset.data_split['train']['x'],
                                self.dataset.data_split['train']['y'],
                                intercept=False
                                )

                    elif method.paradigm!="MTL":
                        fname = os.path.join(method_directory,
                            '{}.mdl'.format(method.__str__()))
                        with open(fname, 'rb') as f:
                            model_trained = pickle.load(f)
                        method.load_params(model_trained)
                        method.model.classes_ = np.unique(self.dataset.data_split['train']['y'][0])
                        method.column_names=self.dataset.column_names
                        # store feature importance to output folder
                        method.feature_names = self.dataset.column_names
                        feat_imp = method.feature_importance()
                        feature_importance = {'Pooled': feat_imp}
                    else:
                        method.feature_names = self.dataset.column_names
                        feat_imp = method.feature_importance()
                        feature_importance = {self.dataset.data_split['task_name'][t]:fi for t, fi in enumerate(feat_imp)}
                    y_pred = method.predict(self.dataset.data_split['test']['x'])
                    results = {}
                    ypred_yobs = {}
                    for t in range(self.nb_tasks):
                        result_task = {}
                        y_true = self.dataset.data_split['test']['y'][t]
                        ypred_yobs[t] = {'pred': y_pred[t],
                                        'obs': y_true}

                        for met in self.metrics:
                            result_task[met] = metric_func[met](y_pred[t],
                                                                y_true)
                        n_samples = self.dataset.data_split['train']['x'][t].shape[0]
                        results[t] = {'results': result_task,
                                        'sample_size': n_samples}
                elif method.paradigm == 'GT':
                    method.fit(self.dataset.data_split['train']['x'],
                            self.dataset.data_split['train']['y'],
                            self.dataset.data_split["weight"],
                            self.dataset.data_split["beta_vec"])

                    y_pred = method.predict(self.dataset.data_split['test']['x'])
                    results = {}
                    ypred_yobs = {}
                    for t in range(self.nb_tasks):
                        result_task = {}
                        y_true = self.dataset.data_split['test']['y'][t]
                        ypred_yobs[t] = {'pred': y_pred[t],
                                         'obs': y_true}
                        for met in self.metrics:
                            result_task[met] = metric_func[met](y_pred[t],
                                                                y_true)
                        n_samples = self.dataset.data_split['train']['x'][t].shape[0]
                        results[t] = {'results': result_task,
                                      'sample_size': n_samples}

                else:
                    raise ValueError('Unknown paradigm %s' % (method.paradigm))

                # save predicted and observed values to file
                output_fname1 = os.path.join(method_directory,
                                            '{}_pred_obs.dat'.format(method.__str__()))
                with open(output_fname1, 'wb') as fh:
                    pickle.dump(ypred_yobs, fh)

                # save all performances to file
                output_fname = os.path.join(method_directory,
                                            '{}.pkl'.format(method.__str__()))
                with open(output_fname, 'wb') as fh:
                    pickle.dump(results, fh)

                # few additional metrics for the simulatsioned dataset
                # 1. Support prediction confusion metrics
                # 2. Recovery of ground truth weight (measure in cosine distance)
                if method.paradigm=='MTL':
                    support_pred = method.return_support()
                    support_pred[support_pred<1e-10]=0
                    support_pred[support_pred>1e-10]=1
                    support_pred = support_pred.astype(np.bool_)
                    beta_vec = self.dataset.data_split["beta_vec"].astype(np.bool_)
                    support = beta_vec*support_pred
                    weights = self.dataset.data_split["weight"]

                    result_support = {}
                    for met in self.metrics:
                        result_support[met] = metric_func[met](support_pred,beta_vec)

                    if method.name=="BayesMTL":
                        weights_est = method.return_params()["m"]*method.return_params()["phi"]
                    else:
                        weights_est = method.W.T
                    result_support["cos distance"] = 1-np.trace(weights_est.T@weights)/(np.linalg.norm(weights_est,'fro')*np.linalg.norm(weights,'fro'))
                    output_fname = os.path.join(method_directory,
                                        '{}_support.supp'.format(method.__str__()))
                    with open(output_fname, 'wb') as fh:
                        pickle.dump(result_support, fh)

                self.logger.info('Results stored in %s' % (output_fname))

    def generate_report(self, data_subtype=None):
        # read results from experiment folder and store it into a dataframe
        df, op_dict = self.__read_experiment_results(data_subtype)
        txt_filename = os.path.join(const.OUTPUT_FOLDER,
                                self.name,
                                data_subtype,
                                '{}_table.csv'.format(self.name))
        df.to_csv(txt_filename)

        # set output pdf name
        pdf_filename = os.path.join(const.OUTPUT_FOLDER,
                                    self.name,
                                    data_subtype,
                                    '{}_report.pdf'.format(self.name))

        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

        # call several plot functions
        # if self.dataset_name!="Simulated":
        self.__tasks_average_std_plot(df, pdf)
        self.__individual_tasks_plot(df, op_dict, pdf)
        # self.__methods_scatter_plot(df, pdf)
        # self.__methods_diff_bars_plot(df, pdf)
        # self.__pooled_performance_plots(df, op_dict, pdf)
        # self.__plot_precision_recall_curve(df, op_dict, pdf)
        # self.__plot_det_curve(df, op_dict, pdf)
        # close pdf file
        pdf.close()

    def __check_inputs(self, dataset, methods, metrics, nb_runs):
        # make sure all inputs have expected values and types
        # assert isinstance(dataset, Dataset)

        # make sure it received a list of methods
        if not isinstance(methods, list):
            methods = list(methods)
        assert len(methods) > 0

        # make sure it received a list of metrics
        if not isinstance(metrics, list):
            metrics = list(metrics)

        # check if all methods are valid (instance of Method class)
        # for method in methods:
        #     assert isinstance(method, Method)

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]
        # check if all metrics are valid (exist in performance_metrics module)
        for metric in metrics:
            assert metric in existing_metrics

        # number of runs has to be larger then 0
        assert nb_runs > 0

    def check_run_completed(self, run_directory):
        """ Check if a particular run has been fully completed. """
        # 1) check if run folder exists
        # 2) check if all methods folder are present
        # 3) for each method's folder check if all files are present
        if not os.path.exists(run_directory):
            return -1
        present_folders = [f for f in os.listdir(run_directory)]
        necessary_ext = ['.fi', '.log', '.pkl', '.dat']
        for method in self.methods:
            # if particular method's folder is not present
            # then the entire experiment execution has not been completed yet
            if method.__str__() not in present_folders:
                return -2
            method_directory = os.path.join(run_directory, method.__str__())
            present_ext = [os.path.splitext(f)[1] for f in os.listdir(method_directory)]
            for nf in necessary_ext:
                if nf not in present_ext: # necessary file is not present
                    return -3
        # all necessary folders/files are present, run is completed
        return 0

    def check_method_completed(self, method_directory):
        """ Check if a particular method at designated run has been fully completed. """
        # 1) check if method folder exists
        # 2) check if all methods folder are present
        # 3) for each method's folder check if all files are present
        if not os.path.exists(method_directory):
            return -1
        # necessary_ext = ['.log', '.pkl', '.dat']
        necessary_ext = ['.log', '.pkl']
        pred_ext = ['.dat']
        present_ext = [os.path.splitext(f)[1] for f in os.listdir(method_directory)]
        for nf in necessary_ext:
            if nf not in present_ext: # necessary file is not present
                return -3
        for nf in pred_ext:
            if nf not in present_ext: # prediction file is not present
                return -2
        # all necessary folders/files are present, run is completed
        return 0

    def __read_experiment_results(self,data_subtype=None):
        """ Read results from an experiment folder (with multiple methods
        results inside) and place it into a data frame structure.

        Args:
            experiment(str): name of the experiment in 'outputs' directory
        """
        experiment_dir = os.path.join(const.OUTPUT_FOLDER, self.name, data_subtype)

        # list that will contain all results information as a table
        # this list will be inserted in a pandas dataframe to become
        # easier to generate plots and latex tables
        result_contents = list()
        # result_supp = list()
        obs_pred = dict()

        for run in next(os.walk(experiment_dir))[1]:
            obs_pred[run] = dict()
            run_dir = os.path.join(experiment_dir, run)
            # iterate over the methods
            # method definition here is "an execution" of a method
            # the same method (let's say Linear Regression) has two instances
            # with different hyper-parameter values, then there will be two
            # 'methods' here (or two entries in the results table)
            for method in next(os.walk(run_dir))[1]:
                method_dir = os.path.join(run_dir, method)
                # get results filename (the one ending with 'pkl')
                resf = [f for f in os.listdir(method_dir) if f.endswith('.pkl')][0]
                with open(os.path.join(method_dir, resf), 'rb') as fh:
                    # dict with each task result as a key
                    # for each key is assigned a dict w/ task specific results
                    tasks_results = pickle.load(fh)
                    for k in tasks_results.keys():
                        # iterate over metrics for k-th task
                        for m in tasks_results[k]['results'].keys():
                            result_contents.append([run, method, method, k, m,
                                                    tasks_results[k]['results'][m],
                                                    tasks_results[k]['sample_size']])
                resf = [f for f in os.listdir(method_dir) if f.endswith('.dat')][0]

                with open(os.path.join(method_dir, resf), 'rb') as fh:
                    tasks_results = pickle.load(fh)
                    obs_pred[run][method] = tasks_results

        # store result_contents list into a dataframe for easier manipulation
        column_names = ['Run', 'Method', 'Methods', 'Task', 'Metric', 'Value', 'SampleSize']
        df = pd.DataFrame(result_contents, columns=column_names)
        # df_supp = pd.DataFrame(result_supp, columns=['Run','Method','Metric','Value'])
        return df, obs_pred

    def __read_feature_importance(self):
        """ Read feature importance information from the pd.Series """

        experiment_dir = os.path.join(const.OUTPUT_FOLDER, self.name)

        # store all pd.Series in a dict structure with method/run/task id
        fi_contents = dict()

        for run in next(os.walk(experiment_dir))[1]:
            fi_contents[run] = dict()
            run_dir = os.path.join(experiment_dir, run)
            # iterate over the methods
            # method definition here is "an execution" of a method
            # the same method (let's say Linear Regression) has two instances
            # with different hyper-parameter values, then there will be two
            # 'methods' here (or two entries in the results table)
            for method in next(os.walk(run_dir))[1]:
                method_dir = os.path.join(run_dir, method)
                if method not in fi_contents[run].keys():
                    fi_contents[run][method] = dict()
                # get feature importance filename (the one ending with '.fi')
                resf = [f for f in os.listdir(method_dir) if f.endswith('.fi')][0]
                with open(os.path.join(method_dir, resf), 'rb') as fh:
                    # dict with each task result as a key
                    # for each key is assigned a dict w/ task specific results
                    tasks_results = pickle.load(fh)
                    for k in tasks_results.keys():
                        if tasks_results[k] is not None:
                            fi_contents[run][method][k] = tasks_results[k]
                        # else:
                            # print(fi_contents[run].keys())
                            # del fi_contents[run][method]
        return fi_contents

    def __tasks_average_std_plot(self, df, pdf):
        """ Plot tasks average and std results.

        It plots a single plot with the average and std results over all tasks.

        Args:
            df (pandas.DataFrame): panda's dataframe containing the results.
            pdf (matplotlib.backend.pdf): matplotlib object to write plots into
                    a pdf file.
            title (string): plot title.
        """
        df.is_copy = None
        runs = df['Run'].unique()
        methods = df['Method'].unique()
        methods_abrrev = df['Methods'].unique()
        tasks = df['Task'].unique()
        metrics = self.metrics

        colors = list(mcolors.CSS4_COLORS)

        for metric in metrics:
            df_m = df.loc[df["Metric"] == metric]
            perf_mat = np.zeros((len(runs), len(methods), len(tasks)))
            for i, r in enumerate(runs):
                for j, m in enumerate(methods):
                    df_ij = df_m.loc[(df_m['Run'] == r) & (df_m['Method'] == m)]
                    perf_mat[i, j, :] = df_ij['Value'].values

            # colors = ['azure', 'green', 'sienna', 'orchid', 'darkblue']
            # colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
            fig, ax1 = plt.subplots()
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # weights = np.log(df_ij["SampleSize"].values)
            # bplot = plt.boxplot(np.average(perf_mat,weights=weights,axis=2), patch_artist=True) # weighted version
            if metric == "mcc":
                bplot = plt.boxplot(perf_mat.mean(axis=2), patch_artist=True)
            else:
                # bplot = plt.boxplot(gmean(perf_mat,axis=2), patch_artist=True) # geometric mean version
                bplot = plt.boxplot(perf_mat.mean(axis=2), patch_artist=True)

            plt.xticks(1+np.arange(len(methods_abrrev)),
                       methods_abrrev, fontsize=10)
            plt.ylabel(' '.join(metric.split('_')).title(), fontsize=12)
            plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

            # for i, (p1, p2) in enumerate(zip(bplot['boxes'], bplot['fliers'])):
            #     # color = mcolors.CSS4_COLORS[colors[i]] #mcolors.to_rgba(mcolors.TABLEAU_COLORS[colors[i]])
            #     # print(color)
            #     p1.set(facecolor=colors[i])
            #     p2.set(color=colors[i])

            pdf.savefig(fig)
            plt.clf()

    def __methods_scatter_plot(self, df, pdf):
        metrics = self.metrics
        for metric in metrics:
            df_s = df.loc[df["Metric"] == metric]
            methods = df_s['Method'].unique()
            for i, met1 in enumerate(methods):
                for j, met2 in enumerate(methods):
                    if i < j:
                        df_s1 = df_s.loc[df["Method"] == met1]
                        df_s2 = df_s.loc[df["Method"] == met2]
                        runs = df_s1['Run'].unique()
                        runs_m1 = np.zeros((self.nb_tasks, len(runs)))
                        runs_m2 = np.zeros((self.nb_tasks, len(runs)))
                        for k,run in enumerate(runs):
                            runs_m1[:, k] = df_s1.loc[df_s1['Run']==run,'Value'].as_matrix()
                            runs_m2[:, k] = df_s2.loc[df_s2['Run']==run,'Value'].as_matrix()
                        mean_m1 = runs_m1.mean(axis=1)
                        mean_m2 = runs_m2.mean(axis=1)
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(mean_m1, mean_m2, 'bo')
                        xmin = min(np.min(mean_m1), np.min(mean_m2))
                        xmax = max(np.max(mean_m1), np.max(mean_m2))
                        ax.plot([xmin*0.8, xmax*1.2],
                                [xmin*0.8, xmax*1.2],
                                ls="--", c=".3")
                        tasks_name = df_s1['Task'].unique()
                        for l, txt in enumerate(tasks_name):
                            ax.annotate(txt[4:], (mean_m1[l],
                                                  mean_m2[l]),
                                        fontsize=8)
                        # ax.set_title('%s: Method X Method' % (metric))
                        ax.set_xlabel(met1.split('_')[0])
                        ax.set_ylabel(met2.split('_')[0])
                        pdf.savefig(fig)

    def __methods_diff_bars_plot(self, df, pdf):
        fig = plt.figure()
        metrics = self.metrics
        for metric in metrics:
            df_s = df.loc[df["Metric"] == metric]
            methods = df_s['Method'].unique()
            for i, met1 in enumerate(methods):
                for j, met2 in enumerate(methods):
                    if i != j:
                        df_s1 = df_s.loc[df["Method"] == met1]
                        df_s2 = df_s.loc[df["Method"] == met2]

                        runs = df_s1['Run'].unique()
                        runs_m1 = np.zeros((self.nb_tasks, len(runs)))
                        runs_m2 = np.zeros((self.nb_tasks, len(runs)))
                        for k, run in enumerate(runs):
                            if k == 0:
                                ssize = df_s2.loc[df_s2['Run']==run,'SampleSize'].as_matrix()
                            runs_m1[:, k] = df_s1.loc[df_s1['Run']==run,'Value'].as_matrix()
                            runs_m2[:, k] = df_s2.loc[df_s2['Run']==run,'Value'].as_matrix()
                        s_ids = np.argsort(ssize)

                        ax = fig.add_subplot(1, 1, 1)
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                        # relative performance
                        delta = np.divide((runs_m1 - runs_m2), runs_m1, out=np.zeros_like(runs_m1), where=runs_m1!=0) * 100
                        delta = np.minimum(np.maximum(delta, -MAX_REL_IMPROVEMENT), MAX_REL_IMPROVEMENT)
                        delta_mean = delta.mean(axis=1)
                        delta_mean = delta_mean.T[s_ids].T  # reorder columns
                        if metric in THE_HIGHER_THE_BETTER:
                            colors = ['g' if d >= 0 else 'r' for d in delta_mean]
                        else:
                            colors = ['g' if d <= 0 else 'r' for d in delta_mean]

                        barp = plt.bar(np.arange(len(delta_mean)),
                                       delta_mean, color=colors,
                                       yerr=delta.std(axis=1),
                                       error_kw=dict(lw=1))

                        # Add counts above the two bar graphs
                        for k, rect in enumerate(barp):
                            plt.text(rect.get_x() + rect.get_width()/2.0,
                                     0, '%d' % ssize[s_ids[k]],
                                     ha='center', va='bottom', fontsize=10,
                                     color='k')

                        xticks = [xt for xt in df_s1['Task'].unique().tolist()]
                        # xticks = [t.replace('_', '\n') for t in xticks]

                        xticks = np.array(xticks)
                        xi = np.arange(delta.shape[0])
                        plt.xticks(xi, xticks[s_ids])
                        locs, labels = plt.xticks()
                        plt.setp(labels, rotation=90)
                        ax.tick_params(axis='x', which='major', labelsize=6)

                        title_txt = '{} vs {}'.format(met1.split('_')[0],
                                                      met2.split('_')[0])
                        ax.set_title(title_txt, fontsize=20)
                        metric_name = ' '.join(metric.split('_')).title()
                        ax.set_ylabel('{} \n Relative performance (%)'.format(metric_name), fontsize=15)

                        #plt.tight_layout()
                        pdf.savefig(fig)
                        plt.clf()

    def __pooled_performance_plots(self, df, op_dict, pdf):
        """ Compute performance metric of all test samples regardless the task.
        Data from all tasks are pooled and then the performance is computed. """
        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        tasks = df['Task'].unique()
        runs = op_dict.keys()
        methods = op_dict[list(runs)[0]].keys()

        colors = ["windows blue", "amber",
                  "greyish", "faded green", "dusty purple"]

        for i, metric in enumerate(self.metrics):
            perform = list()
            for r, run in enumerate(runs):
                for j, method in enumerate(methods):
                    # accumulate performance from all tasks
                    accum_pred = np.array([])
                    accum_observed = np.array([])
                    accum_v = np.array([])
                    accum_d = np.array([])
                    for task in tasks:
                        accum_pred = np.concatenate((accum_pred, op_dict[run][method][task]['pred'].ravel()))
                        accum_observed = np.concatenate((accum_observed, op_dict[run][method][task]['obs'].ravel()))
                        # accum_v = np.concatenate((accum_v, op_dict[run][method][task]['svv_time'].ravel()))
                        # accum_d = np.concatenate((accum_d, op_dict[run][method][task]['censor_flag'].ravel()))
                    # compute performance from data from all tasks
                    v = metric_func[metric](accum_pred,
                                            accum_observed,
                                            censor_flag=accum_d,
                                            survival_time=accum_v)
                    perform.append([metric, run, method, v])

            column_names = ['Metric', 'Run', 'Method', 'Value']
            df_perf = pd.DataFrame(perform, columns=column_names)
            with sns.plotting_context("notebook", font_scale=1.3):
                g = sns.catplot(x="Method", y="Value", data=df_perf,
                                kind="bar", palette=sns.xkcd_palette(colors))
                g.set_axis_labels("", ' '.join(metric.split('_')).title()) #.set(ylim=(0.7, 0.8))
                plt.tight_layout()
            pdf.savefig(g.fig)
            plt.clf()

    def __individual_tasks_plot(self, df, op_dict, pdf):
        """ Plot results for each task separately.

        It plots results for all metrics for each task separately.
        Each task will have its own plot and page in the pdf report file.

        Args:
            df (pandas.DataFrame): results data frame
            pdf (matplotlib.backend.pdf): matplotlib object to write plots into
                    a pdf file.
        """
        for metric in df['Metric'].unique():

            dfa = df[df['Metric'] == metric].copy()

            for task in df['Task'].unique():

                # dfa2 = dfa[dfa['Task'] == task].copy()

                fig, ax = plt.subplots(1, 1)
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                # Draw a nested boxplot to show bills by day and time
                g = sns.boxplot(x="Method", y="Value",
                                # hue="Method", #palette=["m", "g"],
                                data=dfa[dfa['Task'] == task], ax=ax)
                # g.despine(offset=10, trim=True)
                nb_samples = dfa[dfa['Task'] == task]['SampleSize'].values[0]
                # g.set_title("{} ({} training samples)".format(task, nb_samples))
                g.set_title("{}".format(task))
                g.set_ylabel(metric.title())
                g.set_xlabel('')

                pdf.savefig(fig)

                plt.clf()

    def __feature_importance_report(self, feat_imp, pdf, top_n=10):
        """ Plot the most important features identified by each model. """
        runs = list(feat_imp.keys())
        methods = list(feat_imp[runs[0]].keys())

        for method in methods:

            tasks = list(feat_imp[runs[0]][method].keys())

            for task in tasks:
                fig, ax = plt.subplots() # TODO: adjust for multiple tasks
                all_dfs = list()
                for run in runs:
                    all_dfs.append(feat_imp[run][method][task])
                df = pd.concat(all_dfs, axis=1)

                # get the name of the 'top_n' most important features
                dfs = df.mean(axis=1).apply(np.abs).sort_values(ascending=False)
                top_n_names = dfs.head(top_n).index.to_list()

                # create a plot with mean and std of the feature importances
                df = df.stack().reset_index()
                df.rename(columns={'level_0': 'feature',
                                   'level_1': 'run',
                                   0: 'importance'}, inplace=True)
                df_sel = df[df['feature'].isin(top_n_names)].copy()
                # print(df_sel.groupby('feature')['importance'].agg(np.mean).sort_values(ascending=False))
                sns.barplot(x="importance", y="feature", data=df_sel,
                            order=top_n_names, ax=ax)

                plt.title('%s - %s ' % (method, task))
                plt.xlabel('Importance')
                pdf.savefig(fig)
                plt.clf()
