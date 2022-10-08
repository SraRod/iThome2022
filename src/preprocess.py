import os
import yaml
import torch
import monai
import argparse
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Text, List, Dict


SPLITS = ['TRAIN', 'VALIDATION', 'TEST']
BATCH_SIZE = 16
LABEL_LIST = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural', 'hernia']

def npz_dataset_to_dicts(npz_files: str, splits: str) -> Dict:
    split_mapping = {
        'TRAIN' :'train',
        'VALIDATION' : 'val',
        'TEST' : 'test'}
    sliced_data = {}
    for key in npz_files.files:
        sliced_data[key] = np.split(npz_files[key], len(npz_files[key]))
    del npz_files
    datasets = {
        split : [
            { 
                'img' : sliced_data[f'{split_mapping[split]}_images'][i].transpose(0,2,1),
                'labels' : sliced_data[f'{split_mapping[split]}_labels'][i][0],
            }
            for i in range(len(sliced_data[f'{split_mapping[split]}_images']))
        ]
        for split in SPLITS}
    return datasets


def prepare_dataset(CONFIG: Dict) -> Dict:
    
    dataset_source = os.path.splitext(CONFIG['preprocess']['dataset_file'])[-1]
    if dataset_source == '.csv':
        df = pd.read_csv(CONFIG['preprocess']['dataset_file'])
        datasets = {split : df[df['split'] == split].to_dict('records') for split in CONFIG['train']['splits']}
    elif dataset_source == '.npz':
        npz_files = np.load(CONFIG['preprocess']['dataset_file'])
        datasets = npz_dataset_to_dicts(npz_files, CONFIG['train']['splits'])
        
    return datasets

def prepare_transform(CONFIG: Dict) -> monai.transforms.transform:
    
    dataset_source = os.path.splitext(CONFIG['preprocess']['dataset_file'])[-1]
    if dataset_source == '.csv':
        transforms = [
            monai.transforms.LoadImageD(keys = ['img']),
            monai.transforms.EnsureChannelFirstD(keys = ['img']),
            monai.transforms.ScaleIntensityD(keys = ['img']),
            monai.transforms.ToTensorD(keys = ['img'] + LABEL_LIST),
            monai.transforms.AddChanneld(keys = LABEL_LIST),
            monai.transforms.ConcatItemsd(keys = LABEL_LIST, name = 'labels'),]
            
    elif dataset_source == '.npz':
        transforms = [
            monai.transforms.ScaleIntensityD(keys = ['img']),
            monai.transforms.ToTensorD(keys = ['img']),
        ]
        
    if CONFIG['preprocess']['input_size'] != [28, 28]:
        transforms += [monai.transforms.ResizeWithPadOrCropd(keys = ['img'], spatial_size = CONFIG['preprocess']['input_size'])]
        
    if CONFIG['preprocess']['input_channels'] != 1:
        transforms += [monai.transforms.RepeatChannelD(keys = ['img'], repeats = CONFIG['preprocess']['input_channels'])]

    if CONFIG['preprocess']['imagenet_statistics']:
        imagenet_statistics_convertor = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms += [monai.transforms.Lambdad(keys = ['img'], func = imagenet_statistics_convertor)]
    
    transforms = monai.transforms.Compose(transforms)
    
    return transforms

def prepare_data_aug(CONFIG: Dict) -> monai.transforms.transform:
    
    prob = CONFIG['train']['data_aug']['prob']
    transforms_data_aug = [
        monai.transforms.RandAffined(keys = ['img'], 
                                     prob=prob, 
                                     rotate_range = [-0.1, 0.1], 
                                     translate_range = [-3, 3], 
                                     scale_range = 0.1,
                                     padding_mode = 'zeros'),
        monai.transforms.RandGaussianNoised(keys = ['img'], 
                                            prob=prob, 
                                            mean=0.0, std=0.1),
        monai.transforms.RandFlipd(keys = ['img'],
                                   prob=prob, 
                                   spatial_axis = [0])]
    
    transforms_data_aug = monai.transforms.Compose(transforms_data_aug)
    
    return transforms_data_aug


if __name__ == '__main__':
    
    # read configuration
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    CONFIG = yaml.safe_load(open(args.config))
    
    # build dataset
    datasets = prepare_dataset(CONFIG)
    transforms = prepare_transform(CONFIG)
    
    processed_datasets = {
        split : monai.data.Dataset(data = datasets[split], transform = transforms)    
        for split in SPLITS
    }
    
    if CONFIG['train']['data_aug']['use']:
        transforms_data_aug = prepare_data_aug(CONFIG)
        processed_datasets['TRAIN'] = monai.data.Dataset(data = processed_datasets['TRAIN'], transform = transforms_data_aug)    
    
    data_generators = {
        split : torch.utils.data.DataLoader(processed_datasets[split],
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            collate_fn = monai.data.utils.pad_list_data_collate,
                                            pin_memory=torch.cuda.is_available())
        for split in SPLITS
    }
    
    # sampling dataloader and drawing
    for split in SPLITS:
        for batch in data_generators[split]:
            break;
        
        plt.figure(figsize = (10, 10))
        plt.suptitle(f'{split} Set', y = 0.92 ,fontsize=16)
        for i in range(4):
            for j in range(4):
                plt.subplot(4,4,i*4+j+1)
                plt.imshow(batch['img'][i*4+j][0].T, 'gray', vmin = 0, vmax = 1)
                plt.axis('off')
        plt.savefig(f'sampling of {split}.jpeg')
        plt.close()
        
        
    # sampling data augmentation from one sample
    if CONFIG['train']['data_aug']['use']:
        processed_dataset_da = monai.data.Dataset(data = datasets['TRAIN'][:1], transform = transforms) 
        processed_dataset_da = monai.data.Dataset(data = processed_dataset_da, transform = transforms_data_aug)
        data_generator_da = torch.utils.data.DataLoader(processed_dataset_da,
                                            batch_size = 1,
                                            shuffle = True,
                                            collate_fn = monai.data.utils.pad_list_data_collate,
                                            pin_memory=torch.cuda.is_available())
        
        plt.figure(figsize = (10, 10))
        plt.suptitle(f'Data Augmentation Sampling', y = 0.92 ,fontsize=16)
        for i in range(16):
            plt.subplot(4,4,i+1)
            for batch in data_generator_da:
                plt.imshow(batch['img'][0][0].T, 'gray', vmin = 0, vmax = 1)
            plt.axis('off')
        plt.savefig(f'sampling of data augmentation.jpeg')
        plt.close()
        
            
    
