import os
import yaml
import torch
import monai
import argparse
import torchinfo
import torchmetrics
import pandas as pd
import pytorch_lightning as pl

from typing import Text, List, Dict, Any


def get_backbone(CONFIG: Dict):
    if CONFIG['train']['backbone'] == 'efficientnet-b0':
        monai.networks.nets.efficientnet.efficientnet_params['efficientnet-b0'] = (1, 1, CONFIG['preprocess']['input_size'][0], 0.2, 0.2)
        model = monai.networks.nets.EfficientNetBN('efficientnet-b0', 
                                                   spatial_dims = 2,
                                                   in_channels = CONFIG['preprocess']['input_channels'],
                                                   num_classes = 14,
                                                   pretrained = CONFIG['train']['pretrain'])
    elif 'resnet' in CONFIG['train']['backbone']:
        model = getattr(monai.networks.nets, CONFIG['train']['backbone'])
        model = model(spatial_dims = 2, 
                      pretrained = False, 
                      n_input_channels = CONFIG['preprocess']['input_channel'],
                      num_classes = 14)
    try:
        return model
    except:
        print('do not support other backbone until now!')
                

def label_smoother(tensor, label_smooth_fact):
    return tensor * (1 - 2 * label_smooth_fact) + label_smooth_fact
        
class MultiLabelsModel(pl.LightningModule):
    """
    Lightning Module of Multi-Labels Classification for ChestMNIST
    """
    # initialize
    def __init__(self, CONFIG : Dict, **kwargs):
        super().__init__()
        self.CONFIG = CONFIG
        self.save_hyperparameters()
        self.backbone = get_backbone(CONFIG)
        self.loss_function = getattr(torch.nn, CONFIG['train']['loss_function'])()
        
        # define metrics collector
        self.metrics_list = ['acc', 'auroc']
        self.metrics_history = { f'{dataset}/{metric}' : [] for dataset in ['train', 'val'] for metric in self.metrics_list}
        
    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first epochs        
        if self.CONFIG['train']['optimizer']['warmup_epochs'] > 0:
            if (self.trainer.global_step < self.CONFIG['train']['optimizer']['warmup_epochs'] * self.trainer.num_training_batches) :
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.CONFIG['train']['optimizer']['warmup_epochs'] * self.trainer.num_training_batches))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.trainer.lr_scheduler_configs[0].scheduler._get_closed_form_lr()[0]
        
    def configure_optimizers(self):
        
        if 'weight_decay' in self.CONFIG['train']['optimizer']:
            opt_lambda = self.CONFIG['train']['optimizer']['weight_decay']
        else:
            opt_lambda = 0
        
        opt = getattr(torch.optim, self.CONFIG['train']['optimizer']['name'])
        opt = opt(params=self.parameters(), 
                  lr = self.CONFIG['train']['optimizer']['learning_rate'],
                  weight_decay = opt_lambda)
        
        if self.CONFIG['train']['optimizer']['scheduler']:
            lr_sdr = getattr(torch.optim.lr_scheduler, self.CONFIG['train']['optimizer']['scheduler']['name'])
            lr_sdr = lr_sdr(opt, **self.CONFIG['train']['optimizer']['scheduler']['params'])
            
            if self.CONFIG['train']['optimizer']['scheduler']['name'] == 'ReduceLROnPlateau':
                return {"optimizer": opt, "lr_scheduler": lr_sdr, "monitor": self.CONFIG['train']['optimizer']['scheduler']['monitor']}
            else:
                return [opt], [lr_sdr]
        else:
            return opt    

    def forward(self, x):
        y = self.backbone(x)
        return y

    def weight_decay(self):
        reg_loss = 0
        for name, weight in self.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(weight, p = self.CONFIG['train']['weight_decay']['p'])
        return reg_loss

    def step(self, batch: Any):
        inputs, labels = batch['img'].to(self.device, non_blocking=True), batch['labels'].to(self.device, non_blocking=True)
        preds = self.forward(inputs)
        if self.CONFIG['train']['label_smooth'] > 0:
            smooth_label = label_smoother(labels, self.CONFIG['train']['label_smooth'])
            loss = self.loss_function(preds, smooth_label.float())
        else:
            loss = self.loss_function(preds, labels.float())
        return inputs, preds, labels, loss
        
    def training_step(self, batch: Any, batch_idx: int):
        inputs, preds, labels, loss = self.step(batch)
        self.log('train/loss', loss.item(), on_step=True, on_epoch=True, batch_size = inputs.shape[0])
        
        reg_loss = self.weight_decay()
        self.log('train/reg_loss', reg_loss.item(), on_step=True, on_epoch=True, batch_size = inputs.shape[0])
        
        if self.CONFIG['train']['weight_decay']['lambda'] > 0:
            loss += reg_loss * self.CONFIG['train']['weight_decay']['lambda']
            self.log('train/total_loss', loss.item(), on_step=True, on_epoch=True, batch_size = inputs.shape[0])
            
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int):
        inputs, preds, labels, loss = self.step(batch)
        self.log('val/loss', loss.item(), on_step=True, on_epoch=True, batch_size = inputs.shape[0])
        
        reg_loss = self.weight_decay()
        self.log('val/reg_loss', reg_loss.item(), on_step=True, on_epoch=True, batch_size = inputs.shape[0])
        
        if self.CONFIG['train']['weight_decay']['lambda'] > 0:
            loss += reg_loss
            self.log('val/total_loss', loss.item(), on_step=False, on_epoch=True, batch_size = inputs.shape[0])
        
        return {
            'loss' : loss,
            'preds' : preds,
            'labels' : labels
        }
    
    def test_step(self, batch: Any, batch_idx: int):
        inputs, preds, labels, loss = self.step(batch)
        self.log('test/loss', loss.item(), on_step=False, on_epoch=True, batch_size = inputs.shape[0])
        return {
            'loss' : loss,
            'preds' : preds,
            'labels' : labels
        }
    
    def validation_epoch_end(self, validation_step_outputs: List[Any]):
        preds = torch.cat([output['preds'] for output in validation_step_outputs], dim=0).float()
        labels = torch.cat([output['labels'] for output in validation_step_outputs], dim=0).long()
        probs = torch.nn.Sigmoid()(preds)
        
        # compute metrics and log
        acc_score = torchmetrics.functional.accuracy(probs, labels, mdmc_average = 'global')
        auc_score = monai.metrics.compute_roc_auc(probs, labels, average='macro')
        self.log('val/acc', acc_score.item())
        self.log('val/auroc', auc_score.item())
    
    def test_epoch_end(self, validation_step_outputs: List[Any]):
        preds = torch.cat([output['preds'] for output in validation_step_outputs], dim=0).float()
        labels = torch.cat([output['labels'] for output in validation_step_outputs], dim=0).long()
        probs = torch.nn.Sigmoid()(preds)
        self.test_preds = probs.cpu().numpy()
        self.test_labels = labels.cpu().numpy()
        
        # compute metrics and log
        acc_score = torchmetrics.functional.accuracy(probs, labels, mdmc_average = 'global')
        auc_score = monai.metrics.compute_roc_auc(probs, labels, average='macro')
        self.log('test/acc', acc_score.item())
        self.log('test/auroc', auc_score.item())
        


if __name__ == '__main__':
    
    # read configuration
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    CONFIG = yaml.safe_load(open(args.config))
    
    # build lightning-module model
    net = MultiLabelsModel(CONFIG)
    print(torchinfo.summary(net, input_size=(16,1,28,28)))
    
    # test net
    test_input = torch.rand([16,1,28,28]).to(net.device)
    test_output = net(test_input)
    print('net output shape : ',  test_output.shape)

    
    
    
    