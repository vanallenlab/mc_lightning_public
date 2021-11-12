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

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

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

        # wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)

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

class PretrainedResnet50FT_Hosp_DRO_max(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
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

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        # task_accs = torch.Tensor([])
        # task_f1s = torch.Tensor([])
        # task_losses = torch.Tensor([])  

        task_accs = []
        task_f1s = []
        task_losses = []

        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            # print(task_accs)
            # print(task_acc_src)

            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, torch.Tensor([task_f1_src])))
            # task_losses = torch.cat((task_losses, torch.Tensor([task_loss_src])))
            
            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, task_f1_src))
            # task_losses = torch.cat((task_losses, task_loss_src))

            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_num_hosps', len(set(srcs)))
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return max(task_losses)

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_abstain(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.max_val_f1 = 0
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return self.temperature_scale(out)
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label, _ in valid_loader:
                input = input.cuda()
                logits = self(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    def step(self, who, batch, batch_nb):    
            
        x, task_labels, slide_id = batch

        try:
            srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])
        except:
            srcs = []

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        embs = self(x)

        if who == 'train':
            embs = self.dropout(embs)       
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embs)
        
        #Converting logits to probabilities
        sm = torch.nn.Softmax()
        probabilities = sm(task_logits) 
        
        # confidence_region = torch.any(probabilities > self.hparams.confidence_threshold, 1)
        # confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
        
        confidence_region = torch.any(probabilities > self.hparams.confidence_threshold, 1)

        num_confident = len(task_logits[confidence_region])

        if self.hparams.include_all_val == 'False':            

            self.log(who + '_num_confident', num_confident)

            if num_confident > 0:
                task_logits = task_logits[confidence_region]
                task_labels = task_labels[confidence_region]
                
                if len(srcs) > 0: 
                    srcs = srcs[confidence_region.cpu()]
        
        elif self.hparams.include_all_val == 'True':
            if who == 'train': 

                num_confident = len(task_logits[confidence_region])

                self.log(who + '_num_confident', num_confident)

                if num_confident > 0:
                    task_logits = task_logits[confidence_region]
                    task_labels = task_labels[confidence_region]
                    srcs = srcs[confidence_region.cpu()]
        
        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        
        if (who == 'train') and (self.hparams.include_num_confident == 'True') :
            loss += 1/(num_confident + 1)
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'macro')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
        
        if len(srcs) > 0:    
            self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
            self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
            self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
            self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
            self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
            self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
            self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
            self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
            self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
            self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
            self.log(who + '_mean_srcs_task_loss', torch.mean(torch.tensor(task_losses)))
            self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
    
    def validation_epoch_end(self, outputs):
        
        val_f1 = torch.mean(torch.tensor([output['task_f1'] for output in outputs]))
        self.max_val_f1 = max(self.max_val_f1, val_f1)        
        self.log('best_val_f1', self.max_val_f1)
        
        return 

    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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
