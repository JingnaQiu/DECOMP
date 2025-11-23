import dgl
from typing import Dict, Union
import csv
from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import torch
from torch.distributions import Categorical
from tqdm import tqdm
from dgl.data.utils import load_graphs
import warnings
warnings.filterwarnings("ignore")
from histocartography.ml import HACTModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514
convert_7_3_classes = {
    0: 0,  # N - BT
    1: 0,  # PB - BT
    2: 0,  # UDH - BT
    3: 1,  # ADH - AT
    4: 1,  # FEA - AT
    5: 2,  # DCIS - MT
    6: 2,  # IC - MT
}
def print_AL_annotations(AL_annotations, show_number_only=False):
    print(AL_annotations)
    with open(AL_annotations, "r") as rf:
        res = json.load(rf)[0]
        for key, value in res.items():
            print(
                f"{key} ({len(value)} annotated RoIs)-------------------------------------- ")
            if not show_number_only:
                print(value)


def prepare_AL_annotations(AL_annotations, last_AL_annotations=None):
    """at the beginning of each AL cycle, copy the AL annotation from last cycle"""

    if last_AL_annotations:
        # print_AL_annotations(last_AL_annotations)
        with open(last_AL_annotations, "r") as rf:
            res = json.load(rf)
    else:
        res = [{"train": [],
                "val": []}]

    with open(AL_annotations, "w") as wf:
        json.dump(res, wf)


def prepare_candidate_WSIs_json(candidate_WSIs_json, last_candidate_WSIs_json, last_AL_annotations):
    """
    step 1: copy from last_candidate_WSIs_json
        if last_candidate_WSIs_json not existed (i.e., cycle 1):
            create by adding all WSIs
        else:
            copy
    step 2: update candidate_WSIs_json by removing the annotated WSI in last_AL_annotations
    step 3: refill candidate_WSIs_json if all WSIs have been annotated, but remove WSIs that have no more unannotated RoIs

    """
    with open("code/full_select.json", 'r') as f:
        all_RoIs = json.load(f)[0]

    # step 1
    if not last_candidate_WSIs_json:
        candidate_WSIs = {}
        for set_name in ["train", "val"]:
            all_WSIs = [RoI.split("_")[1] for RoI in all_RoIs[set_name]]
            all_WSIs = list(set([f"BRACS_{WSI}" for WSI in all_WSIs]))
            candidate_WSIs.update({set_name: all_WSIs})
    else:
        with open(last_candidate_WSIs_json, "r") as rf:
            candidate_WSIs = json.load(rf)[0]

    if last_AL_annotations:
        with open(last_AL_annotations, 'r') as f:
            anno_RoIs = json.load(f)[0]

        for set_name in ["train", "val"]:
            # step 2
            anno_WSIs = list(set([RoI.split("_")[1] for RoI in anno_RoIs[set_name]]))
            anno_WSIs = [f"BRACS_{WSI}" for WSI in anno_WSIs]
            candidate_WSIs[set_name] = list(set(candidate_WSIs[set_name])-set(anno_WSIs))
        
            # step 3
            if len(candidate_WSIs[set_name]) == 0:
                unanno_RoIs = list(set(all_RoIs[set_name]) - set(anno_RoIs[set_name]))
                unanno_WSIs = [RoI.split("_")[1] for RoI in unanno_RoIs]
                unanno_WSIs = list(set([f"BRACS_{WSI}" for WSI in unanno_WSIs]))
                print(f"restart WSI selections, candidate_WSIs ({set_name}): {len(unanno_WSIs)}")
                candidate_WSIs[set_name] = unanno_WSIs

    with open(candidate_WSIs_json, "w") as wf:
        json.dump([candidate_WSIs], wf)


def get_anno_statics(classes, full_annotations, AL_annotations, return_percentage=True):
    train_val_RoIs_full = np.sum(np.vstack((full_annotations["train"], full_annotations["val"])), axis=0)
    FULL_WSIs = {
        "train": 193,
        "val": 68
    }
    train_val_WSIs_full = np.sum((FULL_WSIs["train"], FULL_WSIs["val"]))


    with open(AL_annotations, 'r') as f:
        f_data = json.load(f)[0]

    print(classes)
    anno_WSIs_list = []
    anno_files_list = []
    anno_files_classes_list = []
    for set_name in ["train", "val"]:
        files = f_data[set_name]
        anno_WSIs = len(set([f.split("_")[1] for f in files]))
        anno_files = len(files)

        if len(classes) == 7:
            anno_files_classes = [np.sum([f.__contains__(c) for f in files]) for c in classes]
        else:
            anno_files_classes = [np.sum([f.__contains__(c) for f in files]) for c in ["N", "PB", "UDH", "ADH", "FEA", "DCIS", "IC"]]
            anno_files_classes = [np.sum(anno_files_classes[:3]), np.sum(anno_files_classes[3:5]), np.sum(anno_files_classes[5:7])]

        print(f"{set_name:8s}: {anno_files:4d} files ({anno_WSIs} WSIs) - {anno_files_classes}")

        anno_WSIs_list.append(anno_WSIs)
        anno_files_list.append(anno_files)
        anno_files_classes_list.append(anno_files_classes)

    anno_WSIs = np.sum(anno_WSIs_list)
    anno_files = np.sum(anno_files_list)
    anno_files_classes = np.sum(np.vstack(anno_files_classes_list), axis=0)
    
    print(f"in total: {anno_files} files - {anno_files_classes}")

    if return_percentage:
        anno_WSIs = np.round(anno_WSIs / train_val_WSIs_full * 100, 2)
        anno_files = np.round(anno_files / np.sum(train_val_RoIs_full) * 100, 2)
        anno_files_classes = np.round(np.array(anno_files_classes) / train_val_RoIs_full * 100, 2)
        print(f"in total: {anno_files}% files - {anno_files_classes}%")

    return anno_WSIs, anno_files, anno_files_classes


def update_class_select_weights(classes, w):
    if isinstance(w, list):
        w = np.array(w)

    if np.sum(w) == 0:
        w = [1 / len(w)] * len(w)
    else:
        w = w / np.sum(w)  # normalize
    w = dict(zip(range(len(classes)), w))
    print("class_select_weights:-----------------------------------------------------------")
    print(classes)
    print([f"{key}: {value:.2f}" for key, value in w.items()])
    return w

