from ast import arg
from email.policy import default
from fcntl import DN_DELETE
from re import A

import argparse
import os
import socket
import sys

import psutil
import setproctitle
import torch.nn
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
# from data_preprocessing.molecule.data_loader import *
import logging
from data_preprocessing.torchdrug.data_loader import ClinToxDataLoader
from training.torchdrug.torchdrug_trainer import TorchDrugTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init

from experiments.distributed.initializer import add_federated_args, get_fl_algorithm_initializer, set_seed
from torchdrug import models, tasks;
from experiments.distributed.torchdrug.before_running import add_args, load_data, create_model, init_training_device, post_complete_message_to_sweep_process


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the process name
    str_process_name = "FedGraphNN:" + str(process_id)
    setproctitle.setproctitle(str_process_name)



    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).


    set_seed(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(
        process_id, worker_number - 1, args.gpu_num_per_server
    )

    # load data
    dataset = load_data(args, args.dataset)
    [
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
    ] = dataset
 
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    
    args.node_embedding_dim = train_data_global.dataset.dataset.node_feature_dim
    model, trainer = create_model(args, args.model, train_data_global)

    return_metrics, best_model_params = trainer.train(train_data_global, device, args, test_data_global)
    for key in return_metrics:
        print(key, return_metrics[key])
 