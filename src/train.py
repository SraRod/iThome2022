import os
import yaml
import tqdm
import torch
import monai
import joblib
import argparse
import torchinfo
import pandas as pd
import pytorch_lightning as pl
from src import model
from src import preprocess

from src.preprocess import SPLITS



if __name__ == '__main__':
    
    # read configuration
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    CONFIG = yaml.safe_load(open(args.config))
    
    # prepare dataset
    df = pd.read_csv(CONFIG['make_dataset']['dataset_file'])
    datasets = {split : df[df['split'] == split].to_dict('records') for split in SPLITS}

    transforms = preprocess.prepare_transform()

    processed_datasets = {
        split : monai.data.Dataset(data = datasets[split], transform = transforms)    
        for split in SPLITS
    }
    data_generators = {
        split : torch.utils.data.DataLoader(processed_datasets[split],
                                            batch_size =  CONFIG['train']['batch_size'],
                                            shuffle = CONFIG['train']['shuffle'][split],
                                            collate_fn = monai.data.utils.pad_list_data_collate,
                                            pin_memory=torch.cuda.is_available(),
                                            **CONFIG['train']['data_loader'])
        for split in SPLITS
    }
    
    # build model
    net = model.MultiLabelsModel(CONFIG)
    print(torchinfo.summary(net, input_size=(16,1,28,28)))
    
    # set callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=CONFIG['train']['weights_folder'],
                                                       monitor= 'val/auroc',
                                                       mode='max',
                                                       save_top_k=3,
                                                       filename = 'epoch_{epoch:02d}_val_loss_{val/loss:.2f}_val_acc_{val/acc:.2f}_val_auroc_{val/auroc:.2f}',
                                                       auto_insert_metric_name = False)
    
    # set trainer
    trainer = pl.Trainer(
        callbacks = checkpoint_callback,
        default_root_dir = CONFIG['train']['weights_folder'],
        max_epochs = CONFIG['train']['max_epochs'],
        limit_train_batches = CONFIG['train']['steps_in_epoch'],
        accelerator = 'cuda',
        devices = 1)
    
    # model training
    trainer.fit(net, 
                data_generators['TRAIN'], 
                data_generators['VALIDATION'])
    