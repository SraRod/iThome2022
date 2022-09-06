import os
import argparse
import medmnist
from typing import Text
from medmnist.info import INFO, DEFAULT_ROOT


DATASET = 'chestmnist'
TEMP = '/tmp'
FOLDER = 'data'
POSTFIX = 'png'


def make_dataset() -> None:
    for split in ["train", "val", "test"]:
        print(f"Saving {DATASET} {split}...")
        dataset = getattr(medmnist, INFO[DATASET]['python_class'])(
            split=split, root=TEMP, download=True)
        dataset.save(FOLDER, POSTFIX)  

if __name__ == '__main__':
    make_dataset()