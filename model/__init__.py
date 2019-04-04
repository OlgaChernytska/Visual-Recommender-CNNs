import torch


def get_model(config):
    '''config is dict with required keys - model and cuda'''
    
    model_name = config['model']
    
    if model_name == 'resnet18':
        from .resnet18 import ResNet18
        model = ResNet18()
        
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    
    model = model.cuda(int(config['cuda']))

    return model
