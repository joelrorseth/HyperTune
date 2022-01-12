#!/usr/bin/env python

import argparse
import json
import sys
import logging
from experiment import GridSearchExperiment, ExperimentConfig
from hyperparameters import HyperparameterSpace
from evaluator import Evaluator
from getpass import getpass

"""
python controller.py \
        --venv /u4/jerorset/cs848/CS848-Project/venv \
        --dnn /u4/jerorset/cs848/CS848-Project/models/ImageNet/train.py  \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/models/ImageNet/hyperparameter_space_ImageNet.json \
        --data /u4/jerorseth/datasets/ILSVRC/Data/CLS-LOC \
        --arch resnet \
        --parallelism mp \
        --epochs 1 \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max \
        --username jerorset \
        --machines gpu1 gpu2 gpu3
"""

supported_model_architectures = ['resnet', 'alexnet']
supported_parallelism_strategies = ['none', 'dp', 'mp', 'gpipe']


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', required=True, type=str, help='Username for SSH to remote machines')
    parser.add_argument('--password', required=False, type=str, help='Password for SSH to remote machines')
    parser.add_argument('--machines', required=True, nargs='+', help='All remote machines to utilize')
    parser.add_argument('--venv', required=True, type=str, help='The venv directory')
    parser.add_argument('--dnn', required=True, type=str, help='The Python file containing the PyTorch DNN training job')
    parser.add_argument('--dnn_hyperparameter_space', required=True, type=str, help='The JSON file defining the DNN hyperparameter space')
    parser.add_argument('--data', required=False, help='path to dataset (if applicable)')
    parser.add_argument('--arch', default='resnet', choices=supported_model_architectures,
        help=f"model architecture: {' | '.join(supported_model_architectures)} (default: resnet)")
    parser.add_argument('--parallelism', default='dp', choices=supported_parallelism_strategies,
        help=f"training parallelism strategy: {' | '.join(supported_parallelism_strategies)} (default: dp)")
    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--dnn_metric_key', required=True, type=str, help='The key for the relevant metric to extract from DNN JSON output')
    parser.add_argument('--dnn_metric_objective', required=True, choices=['max', 'min'], help='Whether to maximize or minimize the metric')
    parser.add_argument('--debug', help="Print all debugging statements", action="store_const",
            dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--verbose', help="Print all logging statements", action="store_const", dest="loglevel", const=logging.INFO)
    
    args = parser.parse_args()
    if not args.password:
        args.password = getpass('Password for SSH to remote machines:')
    return args

def init_hyperparameter_space(path):
    with open(path) as f:
        hyp_dict = json.load(f)
        return HyperparameterSpace(hyp_dict)

def main():
    args = init_args()
    hyperparameter_space = init_hyperparameter_space(args.dnn_hyperparameter_space)

    # Create logger that will be shared with all modules
    logger = logging.getLogger(__name__)
    logger.setLevel(args.loglevel)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setLevel(args.loglevel)
    log_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(log_handler)

    # Run grid search hyperparameter optimization
    experiment_config = ExperimentConfig(args)
    experiment = GridSearchExperiment(experiment_config, hyperparameter_space, logger)
    
    all_trial_results = experiment.run()
    all_printout = "\n".join(str(r) for r in all_trial_results)
    print(f"All Trial Results:")
    print(all_printout)

    evaluator = Evaluator(args.dnn_metric_objective == 'min')
    best_trial_result = evaluator.get_best(all_trial_results) 
    print(f"Best Trial Result:")
    print(str(best_trial_result))


if __name__ == '__main__':
    main()
