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
        monai.networks.nets.efficientnet.efficientnet_params['efficientnet-b0'] = (1, 1, 28, 0.2, 0.2)
        model = monai.networks.nets.EfficientNetBN('efficientnet-b0', 
                                                   spatial_dims = 2,
                                                   in_channels = 1,
                                                   num_classes = 14,
                                                   pretrained=False)
        return model
    else:
        print('do not support other backbone until now!')
                

        
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
        
        # define metrics collecter
        self.metrics_list = ['acc', 'auroc']
        self.metrics_history = { f'{dataset}/{metric}' : [] for dataset in ['train', 'val'] for metric in self.metrics_list}
        
        
    def configure_optimizers(self):
        opt = getattr(torch.optim, self.CONFIG['train']['optimizer'])
        opt = opt(params=self.parameters(), 
                  lr = self.CONFIG['train']['learning_rate'])
        return opt

    def forward(self, x):
        y = self.backbone(x)
        return y

    def step(self, batch: Any):
        inputs, labels = batch['img'].to(self.device), batch['labels'].to(self.device)
        preds = self.forward(inputs)
        loss = self.loss_function(preds, labels.float())
        return inputs, preds, labels, loss
        
    def training_step(self, batch: Any, batch_idx: int):
        inputs, preds, labels, loss = self.step(batch)
        self.log('train/loss', loss.item(), on_step=True, on_epoch=True, batch_size = inputs.shape[0])
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int):
        inputs, preds, labels, loss = self.step(batch)
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, batch_size = inputs.shape[0])
        return {
            'loss' : loss,
            'preds' : preds,
            'labels' : labels
        }
    
    def validation_epoch_end(self, validation_step_outputs: List[Any]):
        preds = torch.cat([output['preds'] for output in validation_step_outputs], dim=0)
        labels = torch.cat([output['labels'] for output in validation_step_outputs], dim=0)
        probs = torch.nn.Sigmoid()(preds)
        
        # compute metrics and log
        acc_score = torchmetrics.functional.accuracy(probs, labels, mdmc_average = 'global')
        auc_score = monai.metrics.compute_roc_auc(probs, labels, average='macro')
        self.log('val/acc', acc_score.item())
        self.log('val/auroc', auc_score.item())
        


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
    
    
    
    