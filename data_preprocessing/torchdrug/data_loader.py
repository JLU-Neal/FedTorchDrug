import copy
import logging
import os
import pickle
import random
from math import log2

import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data

from FedML.fedml_core.non_iid_partition.noniid_partition import (
    partition_class_samples_with_dirichlet_distribution,
)
from torchdrug.datasets import ClinTox


class ClinToxDataLoader:
    def __init__(self, path):
        self.path = path

    def get_data(self):
        dataset = ClinTox(self.path)


    def partition_data_by_sample_size(
        args, path, 
        client_number, 
        uniform=True, 
        compact=True
    ):

    def load_partition_data(
        self,
        args,
        path,
        client_number,
        uniform=True,
        global_test=True,
        compact=True,
        normalize_features=False,
        normalize_adj=False,
    ):
        global_data_dict, partition_dicts = partition_data_by_sample_size(
            args, path, client_number, uniform, compact=compact
        )

        return (
            train_data_num,
            val_data_num,
            test_data_num,
            train_data_global,
            val_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            val_data_local_dict,
            test_data_local_dict,
        )
