#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import json
import importlib

from bayesmtl.utils.design import ModelTraining

import os

modules_dict = {
    "PooledLogisticClassifier" : "bayesmtl.methods.pooled.PooledLogistic",
    "LogisticClassifier" : "bayesmtl.methods.stl.logistic",
    "MSSLClassifier": "bayesmtl.methods.mtl.MSSL",
    "GroundTruthClassifier": "bayesmtl.methods.gt.GT",
    "JFLMTLClassifier": "bayesmtl.methods.mtl.JFLMTL",
    "BayesMTLClassifier" : "bayesmtl.methods.mtl.BayesMTL",
}

data_types = ["unbalanced","balanced"]
data_subtypes = ["Ultra sparse","Sparse","Dense"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--menu', help='Experiment config file.', default="bayesmtl/config/experiment.json")
    parser.add_argument('-N', '--num-cpus',
                        help='Number of CPUs for hyper-parameter selection.',
                        type=int, default=4)
    args = parser.parse_args()
    with open(args.menu, 'r') as fh:
        menu = json.load(fh)

    # creat the set of model classes to be tested
    models_classes = []
    models_params = []
    models_names = []
    for meth_i in menu['methods']:
        submodule = modules_dict[meth_i['model']]
        model_class = getattr(importlib.import_module(submodule), meth_i['model'])
        models_classes.append(model_class)
        models_params.append(meth_i['params'])  # pass model hyper-parameters
        models_names.append(meth_i['name']) # pass model names


    # instantiate dataset object
    submodule = "bayesmtl.data." + menu['data']['dataset']
    class_ = getattr(importlib.import_module(submodule), menu['data']['dataset'])
    data_type = menu["data"]['datatype']
    assert data_type in data_types, \
                     'Unknown data type {}.'.format(data_type)
    nb_runs = menu["training"]['number_runs']
    data_path = os.path.join(os.getcwd(),"bayesmtl","data","simulated",data_type)

    # list of metrics: measure methods' performance
    # see list of available metrics in utils/performance_metrics.py
    metrics = menu['training']['metrics']

    training_config = menu['training']
    print("Resume training: {}".format(training_config['resume_training']))
    breakpoint()
    for data_subtype in data_subtypes:
        exp = ModelTraining("{}-{}-{}".format(menu['training']['exp_name'],
                                            menu['data']['dataset'],
                                            menu['data']['datatype']),
                            menu['data']['dataset'])
        exp.execute(training_config, None, models_classes,
                models_params, method_names=models_names,
                num_cpus=args.num_cpus,resume_training=training_config['resume_training'],
                data_subtype = data_subtype, data_path=data_path, data_class=class_)
        exp.generate_report(data_subtype = data_subtype)
