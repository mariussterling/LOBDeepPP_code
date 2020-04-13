#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:39:10 2019

@author: Marius Sterling
"""
import re
from argparse import ArgumentParser
from itertools import chain
from LOBDeepPP.LOBDeepPP_model import LOBDeepPP_models
from LOBDeepPP.LOBDeepPP_keras_train_class import LOBDeepPP_keras_train_class

# %% parameter selection
parser = ArgumentParser()
parser.add_argument(
    "-t", "--test", action='store_true', dest="test", default=False,
    help="testing setting, by running training and validation on subset")
parser.add_argument(
    "-v", "--verbose", type=int, dest="verbose", default=0,
    help="verbose")
parser.add_argument(
    '--path_params', dest='path_params',
    default='LOBDeepPP/LOBDeepPP_params_files/params_L10a.json',
    help='path to options file. Default:'
    + 'LOBDeepPP/LOBDeepPP_params_files/params_L10a.json')
poss_models = sorted([i for i in dir(LOBDeepPP_models) if 'lob_keras_model'
                      in i.lower()])
poss_models = sorted(poss_models, key=lambda x: int(re.sub(
    "[^0-9]", "", x.replace('LOB_keras_model', '')[:2])))
poss_models_lower = [i.lower() for i in poss_models]
poss_models_short = [i.replace('LOB_keras_model', '') for i in poss_models]

parser.add_argument(
    '--model', type=str, default=poss_models[0], dest='lob_model',
    choices=list(chain.from_iterable([poss_models_short])))
opt = parser.parse_args()

if opt.lob_model in poss_models_short:
    opt.lob_model = dict(zip(poss_models_short, poss_models))[opt.lob_model]
elif opt.lob_model in poss_models_lower:
    opt.lob_model = dict(zip(poss_models_lower, poss_models))[opt.lob_model]

# %% training model based on arguments given
a = LOBDeepPP_keras_train_class(opt.lob_model)
a.set_model_params(opt.path_params)

a.set_up_model()
a.train(opt)
