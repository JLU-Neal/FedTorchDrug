from cgi import test
import logging
from unittest import result
from itertools import islice
import numpy as np
import torch
import torch.nn.functional as F

import wandb

from torch_geometric.utils import negative_sampling

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc

from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from torchdrug import data, utils, tasks
from torchdrug.utils import comm

class TorchDrugTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        if args.scheduler == 'None':
            self.scheduler = None


    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)
        
        
    def train(self, train_data_loader, device, args, test_data_loader=None):
        model = self.model
        # task = tasks.PropertyPrediction(model, task=train_data_loader.dataset.dataset.tasks, criterion="bce", metric=("auprc", "auroc"))
        # if hasattr(task, "preprocess"):
        #     result = task.preprocess(train_data_loader.dataset.dataset, None, None)
        # if model.device.type == "cuda":
        #     task = task.cuda(model.device)
        # model = task
        model.train()
        batch_per_epoch = len(train_data_loader)
    

        if args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        max_test_score = 0
        best_model_params = {}
        for epoch in range(args.epochs):
            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, args.gradient_interval)

            for batch_id, batch in enumerate(
                islice(train_data_loader, batch_per_epoch)
            ):
                if model.device.type == "cuda":
                    batch = utils.cuda(batch, device=model.device)
                
                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)
                print("Epoch = {}, Iter = {}/{}: Train Metric = {}".format(
                        epoch, batch_id + 1, len(train_data_loader), metric
                            )
                )
                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, args.gradient_interval)


                

                if ((batch_id + 1) % args.frequency_of_the_test == 0) or (
                    batch_id == len(train_data_loader) - 1
                ):
                    if test_data_loader is not None:
                        test_metric, _ = self.test(test_data_loader, device, args)
                        print(
                            "Epoch = {}, Iter = {}/{}: Test Metric = {}".format(
                                epoch, batch_id + 1, len(test_data_loader), test_metric
                            )
                        )
                        if test_metric['auroc [CT_TOX]'] > max_test_score:
                            max_test_score = test_metric['auroc [CT_TOX]']
                            best_model_params = {
                                k: v.cpu() for k, v in model.state_dict().items()
                            }
                        print("Current best = {}".format(max_test_score))
            if self.scheduler:
                self.scheduler.step()

        return max_test_score, best_model_params

    def test(self, test_data_loader, device, args):
        logging.info("----------test--------")
        logging.info("len(test_data_loader) = {}".format(len(test_data_loader)))
        model = self.model
        # model_cmp = model
        # task = tasks.PropertyPrediction(model, task=test_data_loader.dataset.dataset.tasks, criterion="bce", metric=("auprc", "auroc"))
        # if hasattr(task, "preprocess"):
        #     result = task.preprocess(test_data_loader.dataset.dataset, None, None)
        # if model.device.type == "cuda":
        #     task = task.cuda(model.device)
        # model = task
        model.eval()
        model.to(device)

        with torch.no_grad():
            
            preds = []
            targets = []
            for batch_id, batch in enumerate(test_data_loader):
                if batch_id == len(test_data_loader) - 1:
                    print(batch)
                if device.type == "cuda":
                    batch = utils.cuda(batch, device=model.device)

                pred, target = model.predict_and_target(batch)
                preds.append(pred)
                targets.append(target)

            pred = utils.cat(preds)
            target = utils.cat(targets)
            metric = model.evaluate(pred, target)
        model.train()
        return metric, model

        


    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metric, model = self.test(test_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(metric['auroc [CT_TOX]'].detach().cpu().numpy())
            logging.info("Client {}, Test ROC-AUC score = {}".format(client_idx, metric['auroc [CT_TOX]']))
            wandb.log({"Client {} CT_TOX/ROC-AUC".format(client_idx): metric['auroc [CT_TOX]']})
        avg_score = np.mean(np.array(score_list))
        logging.info("CT_TOX/ROC-AUC Score = {}".format(avg_score))
        wandb.log({"CT_TOX/ROC-AUC": avg_score})
        return True


    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(
            model_1.state_dict().items(), model_2.state_dict().items()
        ):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismtach found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")