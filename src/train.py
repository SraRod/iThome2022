import os
import torch
import monai
import torchinfo
import pandas as pd
from src import model
from src import preprocess

from typing import Text
from src.preprocess import SPLITS

BATCH_SIZE = 256
MAX_EPOCHS = 1
STEPS_IN_EPOCH = 10


if __name__ == '__main__':
    
    # prepare dataset
    df = pd.read_csv('data/dataset.csv')
    datasets = {split : df[df['split'] == split].to_dict('records') for split in SPLITS}

    transforms = preprocess.prepare_transform()

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
    
    # model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.get_backbone()
    model = model.to(device)
    
    # set training hyperprameters
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    max_epochs = 1

    # training
    epoch_loss_values = []
    for epoch in range(MAX_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in data_generators['TRAIN']:
            step += 1
            inputs, labels = batch['img'].to(device), batch['labels'].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(processed_datasets['TRAIN']) // data_generators['TRAIN'].batch_size}, "
                f"train_loss: {loss.item():.4f}")
            epoch_len = len(processed_datasets['TRAIN']) // data_generators['TRAIN'].batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if step > STEPS_IN_EPOCH:
            break
    