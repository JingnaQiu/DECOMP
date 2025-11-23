import argparse
import os
import pandas as pd
import time
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', type=int, default=1)
    parser.add_argument('--config-path', type=str, default='code/bracs_hact_7_classes_pna_paper.yml', help='train config file path')
    parser.add_argument('--work-dir', type=str, default='results', help='the dir to save logs, models, evaluation results')
    parser.add_argument('--task', type=str, default="3-classes", help='7-class/3-class task')
    parser.add_argument('--cycles', type=int, default=120)

    # AL hyperparameter settings
    parser.add_argument('--init-sampling-strategy', type=str, default='I_random_R_random')

    # parser.add_argument('--image-sampling-strategy', type=str, default='random')
    # parser.add_argument('--image-sampling-strategy', type=str, default='decomposition_threshold_0.7')
    parser.add_argument('--image-sampling-strategy', type=str, default='uncertainty')

    # parser.add_argument('--region-sampling-strategy', type=str, default='random')
    # parser.add_argument('--region-sampling-strategy', type=str, default='decomposition_threshold_0.7')
    # parser.add_argument('--region-sampling-strategy', type=str, default='uncertainty')
    # parser.add_argument('--region-sampling-strategy', type=str, default='uncertainty_clustering')
    # parser.add_argument('--region-sampling-strategy', type=str, default='uncertainty_setcover')
    parser.add_argument('--region-sampling-strategy', type=str, default='BADGE')

    parser.add_argument('--n-query', type=int, default=1, help='the number of selected annotation images per cycle')
    parser.add_argument('--max-query-per-WSI', type=int, default=15, help='the number of selected annotation images per cycle')
    parser.add_argument('--batch_size', type=int, help='batch size', default=16, required=False)
    parser.add_argument('--epochs', type=int, help='epochs', default=60, required=False)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.0005, required=False)

    return parser.parse_args()


def setup():
    args = parse_args()
    args.in_ram = True

    args.data_root = 'path_to_BRACS/'
    args.csv_root = 'path_to_store_results'

    args.csv_root = os.path.join(args.csv_root, args.task, f"n_query_{args.n_query}_max_{args.max_query_per_WSI}")
    os.makedirs(args.csv_root, exist_ok=True)

    args.cg_path = args.data_root + "cell_graphs/"
    args.tg_path = args.data_root + "tissue_graphs/"
    args.assign_mat_path = args.data_root + "assignment_matrices/"

    args.sampling_strategy = f"I_{args.image_sampling_strategy}_R_{args.region_sampling_strategy}"
    args.exp_name = f'exp_{args.exp_id}_{args.sampling_strategy}_n_query_{args.n_query}_max_{args.max_query_per_WSI}'
    args.work_dir = os.path.join(args.work_dir, args.task, f"n_query_{args.n_query}_max_{args.max_query_per_WSI}", args.exp_name)
    os.makedirs(os.path.abspath(args.work_dir), exist_ok=True)

    # load config file
    with open(args.config_path, 'r') as f:
        args.model_config = yaml.safe_load(f)

    if args.task == "3-classes":
        # benign tumors, atypical tumors, malignant tumors
        args.classes = ["BT", "AT", "MT"]
        args.full_annotations = {
            "train": [1231, 1004, 928],
            "val": [261, 162, 179]
        }
    else:
        args.classes = ["N", "PB", "UDH", "ADH", "FEA", "DCIS", "IC"]
        args.full_annotations = {
            "train": [342, 586, 303, 405, 599, 562, 366],
            "val": [86, 87, 88, 77, 85, 97, 82]
        }
    
    args.n_classes = len(args.classes)
    return args
