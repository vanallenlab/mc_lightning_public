import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
import wandb 
import math
from torchmetrics import SpearmanCorrcoef
import torchmetrics


class PretrainedResnet50FT(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--dropout', type=float, default=0.2)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1)
        out = self.dropout(out)     
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch
        
        #Av labels
        self.log(who + '_av_label', torch.mean(task_labels.float()))

        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "mean")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_loss', loss)
        self.log(who + '_acc', task_acc)
        self.log(who + '_f1', task_f1)

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)

        return loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def groupby_agg_mean(self, metric, labels):
        """
        https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/2
        """
        labels = labels.unsqueeze(1).expand(-1, metric.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        #res = torch.zeros_like(unique_labels, dtype=metric.dtype).scatter_add_(0, labels, metric)
        res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, metric)
        res = res / labels_count.float().unsqueeze(1)

        return res

