import argparse
import yaml
import shutil 
import os
import numpy as np
import torch

from dataloader import get_dataloader
from model import get_model



def create_embeddings(config):
    
    exper_folder = config['dir']
    embed_folder = config['embed_folder']
    
    config_file = 'experiment/' + exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    
    dataloader = get_dataloader(config, scope='test')
    model = get_model(config)

    model_weights = 'experiment/' + exper_folder + '/' + config['weights']
    model.load_state_dict(torch.load(model_weights)['model'])

    if os.path.exists(embed_folder):
        shutil.rmtree(embed_folder)
    os.makedirs(embed_folder)
    
    
    with torch.no_grad():
        for num_batch, sample in enumerate(dataloader):
            print('Batch {}/{} processing...'.format(num_batch+1, len(dataloader)))
            
            sample['anchor'] = sample['anchor'].cuda(0)
            embeddings = model(sample['anchor'])
            embeddings = embeddings.detach().cpu().numpy()

            for num in range(len(sample['name'])):
                name = sample['name'][num].replace('/','-')
                name = os.path.join(embed_folder, name)
                vector = embeddings[num]
                np.save(name, vector)

            


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    parser.add_argument('embed_folder', help='Provide embedding folder')
    args = parser.parse_args()
    
    print('Embeddings creating for experiment {}...'.format(args.exper_folder))
    
    config_file = 'experiment/' + args.exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
   
    config['dir'] = args.exper_folder
    config['embed_folder'] = args.embed_folder
    create_embeddings(config)
    
    print('Embeddings created.')