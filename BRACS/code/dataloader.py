"""BRACS Dataset loader."""
import os
import h5py
import json
import shutil
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 

from histocartography.utils import set_graph_on_cuda
from utils import convert_7_3_classes

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}

def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class BRACSDataset(Dataset):
    """BRACS dataset."""

    def __init__(
            self,
            cg_path: str = None,
            tg_path: str = None,
            assign_mat_path: str = None,
            load_in_ram: bool = False,
            task_3_classes: bool = False
    ):
        """
        BRACS dataset constructor.

        Args:
            cg_path (str, optional): Cell Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            tg_path (str, optional): Tissue Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
            task_3_classes (bool, optional): convert 7 classes to 3 classes task
        """
        super(BRACSDataset, self).__init__()

        assert not (cg_path is None and tg_path is None), "You must provide path to at least 1 modality."

        self.cg_path = cg_path
        self.tg_path = tg_path
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram
        self.task_3_classes = task_3_classes

        if cg_path is not None:
            self._load_cg()

        if tg_path is not None:
            self._load_tg()

        if assign_mat_path is not None:
            self._load_assign_mat()

        if self.task_3_classes and self.load_in_ram:
            self.cell_graph_labels = [convert_7_3_classes[ele] for ele in self.cell_graph_labels]
            self.tissue_graph_labels = [convert_7_3_classes[ele] for ele in self.tissue_graph_labels]
        
    def _load_cg(self):
        """
        Load cell graphs
        """
        self.cg_fnames = glob(os.path.join(self.cg_path, '*.bin'))
        self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(fname) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]

    def _load_tg(self):
        """
        Load tissue graphs
        """
        self.tg_fnames = glob(os.path.join(self.tg_path, '*.bin'))
        self.tg_fnames.sort()
        self.num_tg = len(self.tg_fnames)
        if self.load_in_ram:
            tissue_graphs = [load_graphs(fname) for fname in self.tg_fnames]
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]
            self.tissue_graph_labels = [entry[1]['label'].item() for entry in tissue_graphs]

    def _load_assign_mat(self):
        """
        Load assignment matrices 
        """
        self.assign_fnames = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(fname).float().t()
                    for fname in self.assign_fnames
            ]

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        # 1. HACT configuration
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                tg = self.tissue_graphs[index]
                assign_mat = self.assign_matrices[index]
                assert self.cell_graph_labels[index] == self.tissue_graph_labels[index], "The CG and TG are not the same. There was an issue while creating HACT."
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                cg = cg[0]
                label = label['label'].item()
                if self.task_3_classes:
                    label = convert_7_3_classes[label]
                tg, _ = load_graphs(self.tg_fnames[index])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.assign_fnames[index]).float().t()  # (node_tg, node_cg)

            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat

            return cg, tg, assign_mat, label

        # 2. TG-GNN configuration 
        elif hasattr(self, 'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
                label = self.tissue_graph_labels[index]
            else:
                tg, label = load_graphs(self.tg_fnames[index])
                label = label['label'].item()
                if self.task_3_classes:
                    label = convert_7_3_classes[label]
                tg = tg[0]
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            return tg, label

        # 3. CG-GNN configuration 
        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                label = label['label'].item()
                if self.task_3_classes:
                    label = convert_7_3_classes[label]
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, label

    def __len__(self):
        """Return the number of samples in the BRACS dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a BRACS data loader.
    """

    dataset = BRACSDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader


class BRACSDataset_selective(BRACSDataset):
    def __init__(self, **kwargs):
        self.files = self._get_files()
        super(BRACSDataset_selective, self).__init__(**kwargs)

    def _get_files(self):
        pass


    def _load_cg(self):
        """
        Load cell graphs for annotated RoIs
        """
        self.cg_fnames = [os.path.join(
            self.cg_path, fname + '.bin') for fname in self.files]
        # self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(fname) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]
            print(f"loaded {len(cell_graphs)} cell_graphs")

    def _load_tg(self):
        """
        Load tissue graphs for annotated RoIs
        """
        self.tg_fnames = [os.path.join(
            self.tg_path, fname + '.bin') for fname in self.files]
        # self.tg_fnames.sort()
        self.num_tg = len(self.tg_fnames)
        if self.load_in_ram:
            tissue_graphs = [load_graphs(fname) for fname in self.tg_fnames]
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]
            self.tissue_graph_labels = [entry[1]['label'].item() for entry in tissue_graphs]
            print(f"loaded {len(tissue_graphs)} tissue_graphs")

    def _load_assign_mat(self):
        """
        Load assignment matrices for annotated RoIs
        """
        self.assign_fnames = [os.path.join(
            self.assign_mat_path, fname + '.h5') for fname in self.files]
        # self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [h5_to_tensor(
                fname).float().t() for fname in self.assign_fnames]
            print(f"loaded {len(self.assign_matrices)} assign_matrices")


class BRACSDataset_AL(BRACSDataset_selective):
    def __init__(self, AL_annotations, set_name, load_unannotated=False, **kwargs):
        self.AL_annotations = AL_annotations
        self.set_name = set_name
        self.load_unannotated = load_unannotated
        if self.load_unannotated:
            print("loading the unlabeled set !!!")
        super(BRACSDataset_AL, self).__init__(**kwargs)

    def _get_files(self):
        with open(self.AL_annotations, 'r') as f:
            anno_RoIs = json.load(f)[0][self.set_name]

        if not self.load_unannotated:
            return anno_RoIs
        else:
            candidate_WSIs_json = self.AL_annotations.replace("_select.json", "_candidate_WSIs.json")
            with open(candidate_WSIs_json, 'r') as f:
                unanno_WSIs = json.load(f)[0][self.set_name]

            with open("code/filenames.json", 'r') as f:
                WSI_RoI_dict = json.load(f)[0][self.set_name]
            unanno_RoIs = []
            for WSI in unanno_WSIs:
                WSI_all_RoIs = WSI_RoI_dict[WSI]  # all RoIs for a candidate WSI
                WSI_annotated_RoIs = [RoI for RoI in anno_RoIs if WSI+"_" in anno_RoIs]  # check annotated RoIs from this WSI
                unanno_RoIs.extend(list(set(WSI_all_RoIs) - set(WSI_annotated_RoIs)))  # add only unannotated RoIs from this WSI
            
            return unanno_RoIs

def make_al_data_loader(
        AL_annotations,
        set_name,
        batch_size,
        shuffle=True,
        num_workers=0,
        load_unannotated=False,
        **kwargs
):
    """
    Create a BRACS data loader.
    """

    dataset = BRACSDataset_AL(AL_annotations, set_name,
                              load_unannotated, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )

    return dataloader


class BRACSDataset_RoIs(BRACSDataset_selective):
    def __init__(self, RoIs, **kwargs):
        self.RoIs = RoIs
        super(BRACSDataset_RoIs, self).__init__(**kwargs)

    def _get_files(self):
        return self.RoIs


def make_RoI_loader(
        RoIs,
        batch_size,
        shuffle=False,
        num_workers=0,
        **kwargs
):
    """
    Create a BRACS data loader.
    """

    dataset = BRACSDataset_RoIs(RoIs, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )

    return dataloader

"""
Normal Benign UDH ADH FEA DCIS Invasive Total
Train 342 586 303 405 599 562 366 3163
Validation 86 87 88 77 85 97 82 602
Test 84 85 80 86 99 90 102 626
"""
