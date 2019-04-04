import torch


def get_optimizer(config, model):
    
    '''config is dict with required keys - lr and optimizer'''
    
    lr = config['lr']
    optim_name = config['optimizer']
    
        
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    
    else:
        raise ValueError('Unknown optimizer: {}'.format(optim_name))

    
    return optimizer
