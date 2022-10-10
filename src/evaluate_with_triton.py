import os
import yaml
import tqdm
import torch
import monai
import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from src import model
from src import preprocess
from tritonclient.utils import *


def triton_run(processed_img, module_name, triton_api_path):
    with httpclient.InferenceServerClient(triton_api_path) as client:
        # initial inputs format
        inputs = [
            httpclient.InferInput("input", processed_img.shape, 'FP32')
        ]

        outputs = [
            httpclient.InferRequestedOutput("output"),
        ]

        inputs[0].set_data_from_numpy(processed_img.numpy())


        response = client.infer(module_name,
                                inputs,
                                model_version = '1',
                                request_id=str(1),
                                outputs=outputs)

        result = response.get_response()
        output = response.as_numpy("output")
        return output


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
                                                 batch_size =  CONFIG['triton']['batch_size'],
                                                 shuffle = False,
                                                 collate_fn = monai.data.utils.pad_list_data_collate,
                                                 pin_memory = torch.cuda.is_available(),
                                                 **CONFIG['train']['data_loader'])
    
    # inference with processed data
    labels = []
    preds = []
    for batch in tqdm.tqdm(data_generator, total = len(data_generator)):
        pred = triton_run(batch['img'], CONFIG['triton']['module_name'], CONFIG['triton']['triton_api_path'])
        preds.append(pred)
        labels.append(batch['labels'])
        
    # post process and compute metric
    preds = torch.tensor(np.vstack(preds))
    probs = torch.nn.Sigmoid()(preds)
    labels = torch.cat(labels, dim=0).long()
    auc_score = monai.metrics.compute_roc_auc(probs, labels, average='macro')
    
    print(f'Inference Result : AUC = {auc_score}')
    