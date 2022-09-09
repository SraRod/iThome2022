import os
import torch
import monai
import pandas as pd
import matplotlib.pyplot as plt

from typing import Text


SPLITS = ['TRAIN', 'VALIDATION', 'TEST']
BATCH_SIZE = 16

def prepare_transform() -> monai.transforms.transform:
    
    transforms = [
        monai.transforms.LoadImageD(keys = ['img']),
        monai.transforms.EnsureChannelFirstD(keys = ['img']),
        monai.transforms.ScaleIntensityD(keys = ['img']),
        monai.transforms.ToTensorD(keys = ['img'])
    ]

    transforms = monai.transforms.Compose(transforms)
    
    return transforms


if __name__ == '__main__':
    
    df = pd.read_csv('data/dataset.csv')
    datasets = {split : df[df['split'] == split].to_dict('records') for split in SPLITS}
    
    transforms = prepare_transform()
    
    processed_datasets = {
        split : monai.data.Dataset(data = datasets[split], transform = transforms)    
        for split in SPLITS
    }
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
        
            
    
