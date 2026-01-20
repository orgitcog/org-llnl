
import os
import uuid
import ast
import copy

import numpy as np
from numpy.random import RandomState # Move to default_rng future
                                     # once sklearn update this as
                                     # random state is discontinued
                                     # in numpy
import pickle

import ray
from ray import tune
from ray.tune import TuneConfig
from ray.air import CheckpointConfig,session
from ray.air.config import RunConfig

import optuna
from ray.tune.search.optuna import OptunaSearch

from sklearn.model_selection import train_test_split,LeaveOneOut
import pandas as pd

import mpm.config as config


os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1" # this disable saving all the log files
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

log = np.log
eps = np.finfo(float).eps
logn = lambda z: log(z+eps)
logmod = lambda z: log(z+1)

def negative_logistic_loss(y_hat,y):
    y_hat = y_hat.ravel()
    y = y.ravel()
    epsilon = 1e-15
    y_hat=np.clip(y_hat,epsilon,1-epsilon)
    return np.sum(y* np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def objective_function(config, data, model_class, metric, base_model, k=20, seed=1234):
    """ k-Repeated random sub-sampling validation prediction score. """

    if base_model.name == "BayesMTL":
        config_mod = dict()
        alp0 = np.empty(len(base_model.clustering))
        beta0 = np.empty(len(base_model.clustering))
        v0 = np.empty(len(base_model.clustering))
        for key,val in config.items():
            if "alp0" in key:
                alp0[int(key.split('_')[-1])] = val
            elif "beta0" in key:
                beta0[int(key.split('_')[-1])] = val
            elif "v0" in key:
                v0[int(key.split('_')[-1])] = val
            else:
                config_mod[key] = val
        config_mod["alp0"] = alp0
        config_mod["beta0"] = beta0
        config_mod["v0"] = v0
    score = 0
    supp_score = 0
    X = data[0]
    Y = data[1]
    rng = RandomState(seed)
    for i in range(k):
        # check if we are running multitask model
        # or single task, and split the train-test
        # respectively
        if isinstance(X,list):
            X_tr = []
            X_ts = []
            y_tr = []
            y_ts = []
            for _,(x,y) in enumerate(zip(X,Y)):
                [a.append(b) for a,b in zip([X_tr,X_ts,y_tr,y_ts],train_test_split(x, y, test_size=0.2, stratify=y,random_state=rng))]

        else:
            X_tr, X_ts, y_tr, y_ts  = \
                train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=rng)
        if base_model.paradigm == "MTL":
            model = copy.deepcopy(base_model)
            if base_model.name == "BayesMTL":
                model.set_params(config_mod)
            else:
                model.set_params(config)
        else:
            model = model_class(**config)

        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_ts,prob=True) # use log-loss instead
        if isinstance(y_pred,list):
            for t,(y1,y2) in enumerate(zip(y_pred,y_ts)):
                score += negative_logistic_loss(y1, y2)/len(y1.ravel())
        else:
            score += negative_logistic_loss(y_pred, y_ts)/len(y_pred.ravel())
        if base_model.paradigm == "MTL":
            if base_model.name == "BayesMTL":
                supp_score = i*supp_score/(i+1)+(model.return_support()/model.ndimensions)/(i+1)
            else:
                supp_pred = model.return_support()
                supp_pred[supp_pred<1e-10]=0
                supp_pred[supp_pred>1e-10]=1
                supp_score = i*supp_score/(i+1)+(np.sum(supp_pred)/model.ndimensions)/(i+1)
    if base_model.paradigm == "MTL":
        session.report({"score": score, "supp_score": supp_score})
    else:
        session.report({"score": score})

def select_best_hyperparameters(X, Y, model_class, params,metric, mode, num_samples,base_model,output_dir="",name=None,run=0, k_hp=20, num_cpu=60, t=None):
    """ Perform hyper-parameter selection using Ray/Tune. """

    if base_model.paradigm == "MTL":
        cpu_per_trial = 1 
    else:
        if num_cpu>num_samples:
            cpu_per_trial = num_cpu//num_samples
        else:
            cpu_per_trial = 1
    print("cpu per trial is {}".format(cpu_per_trial),flush=True)
    if base_model.name=="STL-LC":
        txt_filename = os.path.join(output_dir,'{}_hp_tuning_{}.csv'.format(base_model.name,t))
    else:
        txt_filename = os.path.join(output_dir,'{}_hp_tuning.csv'.format(base_model.name))


    # if there exists previous export hyperparameter files
    # load them and return the results
    if base_model.paradigm == "MTL" and os.path.exists(txt_filename):
        print("Found previous export hyperparameter files, loading and return the best parameters...\n",flush=True)
        df = pd.read_csv(txt_filename)

        # Convert the 'config' column from string back to dictionary
        df['config'] = df['config'].apply(ast.literal_eval)

        best_idx = np.argmax(df["score"])
        bench_mark_score = df["score"][best_idx]
        bench_mark_supp = df["supp_score"][best_idx]

        print("the difference allow for score is 0.01",flush=True)
        for i,result in enumerate(df["score"]):
            if (bench_mark_score-result)/abs(bench_mark_score)<0.01 and bench_mark_supp>df["supp_score"][i]:
                bench_mark_supp = df["supp_score"][i]
                best_idx = i
        print("best support score is {}".format(bench_mark_supp),flush=True)
        print("best score is {}".format(df["score"][best_idx]),flush=True)

        best_params=df["config"][best_idx]
        if base_model.name == "BayesMTL":
            config_mod = dict()
            alp0 = np.empty(len(base_model.clustering))
            beta0 = np.empty(len(base_model.clustering))
            v0 = np.empty(len(base_model.clustering))
            for key,val in best_params.items():
                if "alp0" in key:
                    alp0[int(key.split('_')[-1])] = val
                elif "beta0" in key:
                    beta0[int(key.split('_')[-1])] = val
                elif "v0" in key:
                    v0[int(key.split('_')[-1])] = val
                else:
                    config_mod[key] = val
            config_mod["alp0"] = alp0
            config_mod["beta0"] = beta0
            config_mod["v0"] = v0
            return config_mod
        else:
            return best_params
    elif base_model.paradigm == "STL" and os.path.exists(txt_filename):
        print("Found previous export hyperparameter files, loading and return the best parameters...\n",flush=True)
        df = pd.read_csv(txt_filename)

        # Convert the 'config' column from string back to dictionary
        df['config'] = df['config'].apply(ast.literal_eval)
        best_idx = np.argmax(df["score"])

        return df["config"][best_idx]
    else:
        print("Did not find previous hyperparameter files...\n",flush=True)
        ray.shutdown()
        ray.init(num_cpus=num_cpu)
        search_space = dict()
        for k in params['space'].keys():
            v = params['space'][k]
            if base_model.name == "BayesMTL" and k in ["alp0","beta0","v0"]:
                for i in range(len(base_model.clustering)):
                    search_space["{}_{}".format(k,i)] = optuna.distributions.FloatDistribution(low=v[0], high=v[1], log=True)
            elif params['types'][k] == 'categorical':
                search_space[k] = optuna.distributions.CategoricalDistribution(v)
            elif params['types'][k] == 'uniform-continuous':
                search_space[k] = optuna.distributions.FloatDistribution(low=v[0], high=v[1])
            elif params['types'][k] == 'log-uniform-continuous':
                search_space[k] = optuna.distributions.FloatDistribution(low=v[0], high=v[1], log=True)
            elif params['types'][k] == 'uniform-discrete':
                search_space[k] = optuna.distributions.IntDistribution(v)
            else:
                raise ValueError('Unknown hyper-parameter distribution type.')
        if base_model.paradigm == "MTL":
            algo = OptunaSearch(metric=["score","supp_score"], mode=[mode,"min"], space=search_space,seed=1234)
        else:
            algo = OptunaSearch(metric="score", mode=mode, space=search_space)

        ckpt_config = CheckpointConfig(num_to_keep=1,
                                    checkpoint_score_order=mode,
                                    checkpoint_frequency=0)
        run_config = RunConfig(name=name,
                            verbose=0,
                            local_dir=config.RAY_RESULTS,
                            log_to_file=False,
                            checkpoint_config=ckpt_config)
        tune_config = TuneConfig(search_alg=algo,
                                num_samples=num_samples,
                                reuse_actors=True)

        # Start a Tune run and return the best hyper-parameters
        trainable_with_resources = tune.with_resources(tune.with_parameters(objective_function,
                                                data=(X, Y),
                                                model_class=model_class,
                                                metric=metric,
                                                base_model = base_model,
                                                k = k_hp), {"cpu": cpu_per_trial})
        tuner = tune.Tuner(trainable_with_resources,
                           tune_config=tune_config,
                           run_config=run_config)

        results = tuner.fit()
        best_params = results.get_best_result(metric="score", mode=mode).config
        if base_model.paradigm == "MTL":
            result_contents = list()
            for trial in results:
                if trial.metrics["done"]:
                    trial_data = [trial.metrics["score"],trial.metrics["supp_score"],trial.config]
                    result_contents.append(trial_data)
            column_names = ['score','supp_score','config']
            df = pd.DataFrame(result_contents, columns=column_names)
            txt_filename = os.path.join(output_dir,'{}_hp_tuning.csv'.format(base_model.name))
            df.to_csv(txt_filename)

            bench_mark_score = results.get_best_result(metric="score", mode=mode).metrics["score"]
            bench_mark_supp = results.get_best_result(metric="score", mode=mode).metrics["supp_score"]
            for _,result in enumerate(results):
                if (bench_mark_score-result.metrics["score"])/abs(bench_mark_score)<0.01 and bench_mark_supp>result.metrics["supp_score"]:
                    bench_mark_supp = result.metrics["supp_score"]
                    best_params = result.config
        elif base_model.paradigm == "STL":
            result_contents = list()
            for trial in results:
                if trial.metrics["done"]:
                    trial_data = [trial.metrics["score"],trial.config]
                    result_contents.append(trial_data)
            column_names = ['score','config']
            df = pd.DataFrame(result_contents, columns=column_names)
            txt_filename = os.path.join(output_dir,'{}_hp_tuning_{}.csv'.format(base_model.name,t))
            df.to_csv(txt_filename)
        else:
            result_contents = list()
            for trial in results:
                if trial.metrics["done"]:
                    trial_data = [trial.metrics["score"],trial.config]
                    result_contents.append(trial_data)
            column_names = ['score','config']
            df = pd.DataFrame(result_contents, columns=column_names)
            txt_filename = os.path.join(output_dir,'{}_hp_tuning.csv'.format(base_model.name))
            df.to_csv(txt_filename)

        if base_model.name == "BayesMTL":
            config_mod = dict()
            alp0 = np.empty(len(base_model.clustering))
            beta0 = np.empty(len(base_model.clustering))
            v0 = np.empty(len(base_model.clustering))
            for key,val in best_params.items():
                if "alp0" in key:
                    alp0[int(key.split('_')[-1])] = val
                elif "beta0" in key:
                    beta0[int(key.split('_')[-1])] = val
                elif "v0" in key:
                    v0[int(key.split('_')[-1])] = val
                else:
                    config_mod[key] = val
            config_mod["alp0"] = alp0
            config_mod["beta0"] = beta0
            config_mod["v0"] = v0
            return config_mod
        else:
            return best_params