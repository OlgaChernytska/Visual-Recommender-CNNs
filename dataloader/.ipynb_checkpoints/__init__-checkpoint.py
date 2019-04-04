from torch.utils.data import DataLoader

def get_dataloader(config, scope):
    
    '''config is dict with required keys - dataset, augment, batch_size and shuffle'''
    
    dataset = config['dataset']
    augmentation = config['augment']
    
    if scope == 'test':
        drop_last = False
    else:
        drop_last = True
    
    if dataset == 'deepfashion':
        from .deepfashion import DeepFashion
        dataset = DeepFashion({'path': 'data/', 
                               'augment': augmentation,
                               'scope': scope})    
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
        
    
    return DataLoader(dataset,
                      config['batch_size'],
                      shuffle=bool(config['shuffle']), 
                      num_workers=4,
                      drop_last=drop_last)


