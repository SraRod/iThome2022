import os
import yaml
import wandb
import torch
import argparse
import pytorch_lightning as pl
from src import model



if __name__ == '__main__':
    
    # read configuration
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    CONFIG = yaml.safe_load(open(args.config))
    
    
    # build model and load trained model
    net = model.MultiLabelsModel.load_from_checkpoint(CONFIG['evaluate']['weights_path'])
    
    # convert model to deploy format
    os.makedirs(CONFIG['export']['target'], exist_ok=True)
    export_path = os.path.join(CONFIG['export']['target'], f'model.' + CONFIG['export']['format'])
    if CONFIG['export']['format'] == 'onnx':
        net.to_onnx(export_path, 
                    input_sample = torch.rand([1,3,28,28]), 
                    opset_version = 10, # if set to default, some env would show a warning message
                    export_params=True,
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                  'output' : {0 : 'batch_size'}})
        
    # store result to wandb registry
    wandb.init()
    art = wandb.Artifact(CONFIG['base']['project'], type="model")
    art.add_file(export_path)
    wandb.log_artifact(art)
    
    
    