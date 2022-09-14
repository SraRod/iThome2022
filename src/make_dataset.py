import os
import yaml
import glob2
import argparse
import medmnist
import pandas as pd

from typing import Text
from medmnist.info import INFO, DEFAULT_ROOT


def make_dataset(config_path: Text) -> None:
    
    CONFIG = yaml.safe_load(open(config_path))['make_dataset']
    medmnist_csv_file = os.path.join(CONFIG['save_folder'], CONFIG['dataset_from'] + '.csv')
    
    if not os.path.isfile(medmnist_csv_file):
        for split in ["train", "val", "test"]:
            print(f"Saving {CONFIG['dataset_from']} {split}...")
            dataset = getattr(medmnist, INFO[CONFIG['dataset_from']]['python_class'])(
                split=split, root=CONFIG['tmp_folder'], download=True)
            dataset.save(CONFIG['save_folder'], CONFIG['img_subfilename'])  
            
    df = pd.read_csv(medmnist_csv_file, header = None)
    cols = ['split', 'img'] + list(INFO[CONFIG['dataset_from']]['label'].values())
    df = df.rename(columns={ i: col for i, col in enumerate(cols)})
    df['img'] = df['img'].apply(lambda x: os.path.join(f"data/{CONFIG['dataset_from']}/", x))
    
    df.to_csv(CONFIG['dataset_file'], index = False)

if __name__ == '__main__':
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    make_dataset(config_path=args.config)