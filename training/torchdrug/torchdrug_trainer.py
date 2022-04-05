import logging

import numpy as np
import torch
import torch.nn.functional as F

import wandb

from torch_geometric.utils import negative_sampling

from sklearn.metrics import average_precision_score, roc_auc_score

from FedML.fedml_core.trainer.model_trainer import ModelTrainer

class TorchDrugTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)
        
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        test_data = None
        try:
            test_data = self.test_data
        except:
            pass

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        max_test_score = 0
        best_model_params = {}
        for epoch in range(args.epochs):
            for mol_idxs, (adj_matrix, feature_matrix, label, mask) in enumerate(
                train_data
            ):
                # Pass on molecules that have no labels
                if torch.all(mask == 0).item():
                    continue

                optimizer.zero_grad()

                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)

                # Need to check the return type
                logits = model(adj_matrix, feature_matrix)
                loss = criterion(logits, label) * mask
                loss = loss.sum() / mask.sum()

                loss.backward()
                optimizer.step()

                if ((mol_idxs + 1) % args.frequency_of_the_test == 0) or (
                    mol_idxs == len(train_data) - 1
                ):
                    if test_data is not None:
                        test_score, _ = self.test(self.test_data, device, args)
                        print(
                            "Epoch = {}, Iter = {}/{}: Test Score = {}".format(
                                epoch, mol_idxs + 1, len(train_data), test_score
                            )
                        )
                        if test_score > max_test_score:
                            max_test_score = test_score
                            best_model_params = {
                                k: v.cpu() for k, v in model.state_dict().items()
                            }
                        print("Current best = {}".format(max_test_score))

        return max_test_score, best_model_params