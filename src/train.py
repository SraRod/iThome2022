import os
import yaml
import tqdm
import torch
import monai
import joblib
import argparse
import torchinfo
import torchmetrics
import pandas as pd
from src import model as my_model
from src import preprocess

from typing import Text
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
                                            shuffle = True,
                                            collate_fn = monai.data.utils.pad_list_data_collate,
                                            pin_memory=torch.cuda.is_available(),
                                            **CONFIG['train']['data_loader'])
        for split in SPLITS
    }
    
    # model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_model.get_backbone(CONFIG)
    model = model.to(device)
    
    # set training hyperprameters
    loss_function = getattr(torch.nn, CONFIG['train']['loss_function'])()
    optimizer = getattr(torch.optim, CONFIG['train']['optimizer'])
    optimizer = optimizer(model.parameters(), CONFIG['train']['learning_rate'])
    val_interval = 1


    # set init
    MAX_EPOCHS = CONFIG['train']['max_epochs']
    STEPS_IN_EPOCH = CONFIG['train']['steps_in_epoch']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_epoch_loss_values = []
    val_epoch_acc_values = []
    val_epoch_auc_values = []
    metric_values = []
    
    # training
    for epoch in range(MAX_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        pbar = tqdm.tqdm(data_generators['TRAIN'], total = STEPS_IN_EPOCH)
        for batch in pbar:
            step += 1
            inputs, labels = batch['img'].to(device), batch['labels'].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_description(
                f"Training Epoch {step}/{STEPS_IN_EPOCH} "  #{len(processed_datasets['TRAIN']) // data_generators['TRAIN'].batch_size}, "
                f"train_loss: {loss.item():.4f}")
            epoch_len = len(processed_datasets['TRAIN']) // data_generators['TRAIN'].batch_size
            if step >= STEPS_IN_EPOCH:
                break
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            step = 0
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                pbar = tqdm.tqdm(data_generators['VALIDATION'], total = len(processed_datasets['VALIDATION']) // data_generators['VALIDATION'].batch_size)
                for batch in pbar:
                    step += 1
                    val_images, val_labels =  batch['img'].to(device), batch['labels'].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                    pbar.set_description('Validating ...')
                y_prob = torch.nn.Sigmoid()(y_pred)
                loss = loss_function(y_pred, y.float()).item()
                acc_score = torchmetrics.functional.accuracy(y_prob, y, mdmc_average = 'global').item()
                auc_score = monai.metrics.compute_roc_auc(y_prob, y, average='macro').item()
                val_epoch_loss_values.append(loss)
                val_epoch_acc_values.append(acc_score)
                val_epoch_auc_values.append(auc_score)
            if auc_score > best_metric:
                best_metric = auc_score
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join('artefacts', "best_metric_model.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current loss : {loss:.4f} current AUC: {auc_score:.4f}"
                  f" current accuracy: {acc_score:.4f}"
                  f" best AUC: {best_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")

    print(f"train completed, best_metric(AUC): {best_metric:.4f}"
          f"at epoch: {best_metric_epoch}")
    
    joblib.dump((epoch_loss_values, val_epoch_loss_values, val_epoch_acc_values, val_epoch_auc_values), 'artefacts/records.pkl')
        
    