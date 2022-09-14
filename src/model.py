import os
import torch
import monai
import torchinfo


def get_backbone(CONFIG):
    if CONFIG['train']['backbone'] == 'efficientnet-b0':
        monai.networks.nets.efficientnet.efficientnet_params['efficientnet-b0'] = (1, 1, 28, 0.2, 0.2)
        model = monai.networks.nets.EfficientNetBN('efficientnet-b0', 
                                                   spatial_dims = 2,
                                                   in_channels = 1,
                                                   num_classes = 14)
        return model
    else:
        print('do not support other backbone until now!')



if __name__ == '__main__':
    
    model = get_backbone({'train':{'backbone':'efficientnet-b0'}})
    torchinfo.summary(model, input_size=(16,1,28,28))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_input = torch.rand([16,1,28,28]).to(device)
    test_output = model(test_input)
    print(test_output.shape)
    
    