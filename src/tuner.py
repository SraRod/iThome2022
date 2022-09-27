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
import matplotlib.pyplot as plt

from src import model
from src import preprocess




if __name__ == '__main__':
    
    # read configuration
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    CONFIG = yaml.safe_load(open(args.config))
    
    # prepare dataset
    datasets = preprocess.prepare_dataset(CONFIG)
    transforms = preprocess.prepare_transform(CONFIG)
    
    dataset_fun = getattr(monai.data, CONFIG['train']['dataset_fun']['name'])
    processed_dataset = dataset_fun(data = datasets['TRAIN'], transform = transforms, **CONFIG['train']['dataset_fun']['args']) 
    
    data_generator = torch.utils.data.DataLoader(processed_dataset,
                                                 batch_size =  CONFIG['train']['batch_size'],
                                                 shuffle = CONFIG['train']['shuffle']['TRAIN'],
                                                 collate_fn = monai.data.utils.pad_list_data_collate,
                                                 pin_memory = torch.cuda.is_available(),
                                                 **CONFIG['train']['data_loader'])
    # build model
    net = model.MultiLabelsModel(CONFIG)
    print(torchinfo.summary(net, input_size=(16,1,*CONFIG['preprocess']['input_size'])))
    
    # set tunner
    trainer = pl.Trainer(**CONFIG['train']['trainer'])

    # tunner
    lr_finder = trainer.tuner.lr_find(net, data_generator)

    # Plot with Results can be found in
    fig = lr_finder.plot(suggest=True)
    plt.text(x = lr_finder.suggestion(), y = (min(lr_finder.results['loss']) + max(lr_finder.results['loss']))/2, s = f'{lr_finder.suggestion():.4f}')
    fig.savefig('lr_finder.png')