import copy
import logging
import os
import pickle
import random
from math import log2
from re import T

import matplotlib.pyplot as plt
import seaborn as sns
from torchdrug import data

from FedML.fedml_core.non_iid_partition.noniid_partition import (
    partition_class_samples_with_dirichlet_distribution,
)
from torchdrug.datasets import ClinTox, SIDER, BACE, BBBP, Tox21
import torch


class DrugDataLoader:
    def __init__(self, path):
        self.path = path
        self.dataset = None
        # self.dataset = ClinTox(self.path)

    def get_data(self):
        return self.dataset


    def partition_data_by_sample_size(
        self,
        args,
        client_number, 
        uniform=True, 
        compact=True
    ):
        train_set, valid_set, test_set = torch.utils.data.random_split(self.dataset, 
                                        [int(0.8 * len(self.dataset)), 
                                        int(0.1 * len(self.dataset)), 
                                        len(self.dataset) - int(0.8 * len(self.dataset)) - int(0.1 * len(self.dataset))])
        global_data_dict = {
            "train": train_set,
            "valid": valid_set,
            "test": test_set
        }
        if uniform:
            #Split the dataset uniformly for each client
            train_set_clients_size = [int(len(train_set)/client_number)] * (client_number-1)
            train_set_clients_size = [*train_set_clients_size, len(train_set) - sum(train_set_clients_size)]
        
            val_set_clients_size = [int(len(valid_set)/client_number)] * (client_number-1)
            val_set_clients_size = [*val_set_clients_size, len(valid_set) - sum(val_set_clients_size)]

            test_set_clients_size = [int(len(test_set)/client_number)] * (client_number-1)
            test_set_clients_size = [*test_set_clients_size, len(test_set) - sum(test_set_clients_size)]
         
            all_set_clients_size = train_set_clients_size + val_set_clients_size + test_set_clients_size
            all_set_clients = torch.utils.data.random_split(self.dataset, all_set_clients_size)
            train_set_clients = all_set_clients[:client_number]
            val_set_clients = all_set_clients[client_number:2*client_number]
            test_set_clients = all_set_clients[2*client_number:]
        else:
            raise Exception("Not implemented!")

        partition_dicts = [None] * client_number
        for client in range(client_number):
            train_set_client = train_set_clients[client]
            val_set_client = val_set_clients[client]
            test_set_client = test_set_clients[client]

            partition_dict = {
                "train": train_set_client,
                "valid": val_set_client,
                "test": test_set_client
            }
            partition_dicts[client] = partition_dict

        return global_data_dict, partition_dicts

    def load_partition_data(
        self,
        args,
        client_number,
        uniform=True,
        global_test=True,
        compact=True,
    ):
        global_data_dict, partition_dicts = self.partition_data_by_sample_size(
            args, client_number, uniform, compact=compact
        )
        
        train_data_num = len(global_data_dict["train"])
        val_data_num = len(global_data_dict["valid"])
        test_data_num = len(global_data_dict["test"])


        train_data_global = data.DataLoader(
            global_data_dict["train"],
            batch_size=8,
            shuffle=True,
            pin_memory=True,
        )
        val_data_global = data.DataLoader(
            global_data_dict["valid"],
            batch_size=8,
            shuffle=True,
            pin_memory=True,
        )
        test_data_global = data.DataLoader(
            global_data_dict["test"],
            batch_size=8,
            shuffle=False,
            pin_memory=True,
        )


        data_local_num_dict = dict()
        train_data_local_dict = dict()
        val_data_local_dict = dict()
        test_data_local_dict = dict()
        for client in range(client_number):
            train_dataset_client = partition_dicts[client]["train"]
            val_dataset_client = partition_dicts[client]["valid"]
            test_dataset_client = partition_dicts[client]["test"]

            data_local_num_dict[client] = len(train_dataset_client)
            train_data_local_dict[client] = data.DataLoader(
                train_dataset_client,
                batch_size=8,
                shuffle=True,
                pin_memory=True,
            )
            val_data_local_dict[client] = data.DataLoader(
                val_dataset_client,
                batch_size=8,
                shuffle=False,
                pin_memory=True,
            )
            test_data_local_dict[client] = (
                test_data_global
                if global_test
                else data.DataLoader(
                    test_dataset_client,
                    batch_size=8,
                    shuffle=False,
                    pin_memory=True,
                )
            )

            logging.info(
                "Client idx = {}, local sample number = {}".format(
                    client, len(train_dataset_client)
                )
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


class ClinToxDataLoader(DrugDataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = ClinTox(self.path)


class SIDERDataLoader(DrugDataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = SIDER(self.path)

class BACEDataLoader(DrugDataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = BACE(self.path)

class BBBPDataLoader(DrugDataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = BBBP(self.path)

class Tox21DataLoader(DrugDataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = Tox21(self.path)