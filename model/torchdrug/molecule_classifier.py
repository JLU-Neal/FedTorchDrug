import torch
from torchdrug import datasets, core, models, tasks

dataset = datasets.ClinTox("~/FedGraphNN/data/molecule-datasets/")
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)


model = models.GIN(input_dim=dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256], 
                    short_cut=True, batch_norm=True, concat_hidden=True).cuda()


task = tasks.PropertyPrediction(model, task=dataset.tasks, criterion="bce", metric=("auprc", "auroc"))

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, batch_size=1024, gpus=[0])
solver.train(num_epoch=100)

solver.evaluate("valid")

