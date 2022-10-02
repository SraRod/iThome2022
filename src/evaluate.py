import os
import yaml
import tqdm
import torch
import monai
import joblib
import argparse
import torchinfo
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from src import model
from src import preprocess



if __name__ == '__main__':
    
    # read configuration
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    CONFIG = yaml.safe_load(open(args.config))
    
    # prepare test dataset
    datasets = preprocess.prepare_dataset(CONFIG)
    transforms = preprocess.prepare_transform(CONFIG)

    dataset_fun = getattr(monai.data, CONFIG['train']['dataset_fun']['name'])
    processed_dataset = dataset_fun(data = datasets['TEST'], transform = transforms, **CONFIG['train']['dataset_fun']['args']) 
    data_generator = torch.utils.data.DataLoader(processed_dataset,
                                                 batch_size =  CONFIG['evaluate']['batch_size'],
                                                 shuffle = False,
                                                 collate_fn = monai.data.utils.pad_list_data_collate,
                                                 pin_memory = torch.cuda.is_available(),
                                                 **CONFIG['train']['data_loader'])
    
    # build model and load trained model
    net = model.MultiLabelsModel(CONFIG)
    net = net.load_from_checkpoint(CONFIG['evaluate']['weights_path'])
    
    # initialize the Trainer
    trainer = pl.Trainer(**CONFIG['evaluate']['tester'])

    # test the model
    trainer.test(net, dataloaders=data_generator)
    
    