import argparse
import yaml

import warnings
import torch
import numpy as np

from model import get_model
from dataloader import get_dataloader
from optimizer import get_optimizer
from loss import get_loss


class Trainer:
    def __init__(self, config):
 
        self.cuda = int(config['cuda'])
        self.train_dataloader = get_dataloader(config, scope='train')
        self.val_dataloader = get_dataloader(config, scope='val')
        
        self.model = get_model(config)
        try:
            model_weights = 'experiment/' + config['dir'] + '/' + config['weights']
            self.model.load_state_dict(torch.load(model_weights)['model'])
            print('Weigths loaded')
        except:
            print('Weights randomized')

        self.optimizer = get_optimizer(config, self.model)
        self.total_epochs = config['epochs']
        self.batches_per_epoch = config['batches_per_epoch']
        self.val_batches_per_epoch = config['val_batches_per_epoch']
        
        self.exp_name = config['dir']
        self.save_weights_regularity = config['save_weights_regularity']
        self.final_weights_file = 'experiment/' + config['dir'] + '/weights_last.pth' 
        self.log_file = 'experiment/' + config['dir'] + '/logs.csv' 
        self.loss = get_loss(config)
        
    
    def train(self):

        for epoch in range(1, self.total_epochs+1):
            batches = 0
            batch_losses = []

            for sample in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                
                for key in ['anchor','positive','negative']:
                    sample[key] = sample[key].cuda(self.cuda)
            
                anchor_embed = self.model(sample['anchor'])
                positive_embed = self.model(sample['positive'])
                negative_embed = self.model(sample['negative'])
                loss = self.loss(anchor_embed, positive_embed, negative_embed)  
                loss.backward()
                self.optimizer.step()
                
                batch_losses.append(loss.item())
                batches += 1
                if batches >= self.batches_per_epoch:
                    break
                     
            val_batch_losses = self._validate()
            mean_train_loss = np.mean(batch_losses)
            mean_val_loss = np.mean(val_batch_losses)
            print('====Epoch {}. Train loss: {}. Val loss: {}'.format(epoch, mean_train_loss, mean_val_loss))
            
            self._write_logs(epoch, mean_train_loss, mean_val_loss)
            self._save_model(epoch, regularity=self.save_weights_regularity)
            
        return
              
                
    def _validate(self):
        self.model.eval()
        batches = 0
        batch_losses = []
        
        with torch.no_grad():
            for sample in self.val_dataloader:
                
                for key in ['anchor','positive','negative']:
                    sample[key] = sample[key].cuda(self.cuda)
                    
                anchor_embed = self.model(sample['anchor'])
                positive_embed = self.model(sample['positive'])
                negative_embed = self.model(sample['negative'])
                loss = self.loss(anchor_embed, positive_embed, negative_embed) 
                
                batch_losses.append(loss.item())
                batches += 1
                if batches >= self.val_batches_per_epoch:
                    break
                    
        return batch_losses
    
    
    def _write_logs(self, epoch, mean_train_loss, mean_val_loss):
        if epoch==1:
            with open(self.log_file, 'w') as fp:
                fp.write('epoch,train_loss,val_loss\n')
                
        with open(self.log_file, 'a') as fp:
            fp.write('{},{},{}\n'.format(epoch, mean_train_loss, mean_val_loss))                   
        
        return
    
    
    def _save_model(self, epoch, regularity):
        torch.save({'model': self.model.state_dict()}, 
                   self.final_weights_file)
            
        if epoch % regularity == 0:
            torch.save({'model': self.model.state_dict()}, 
                       'experiment/' + self.exp_name + '/weights_' + str(epoch).zfill(3) + '.pth') 
        return
    

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    args = parser.parse_args()
    
    print('Experiment {} started'.format(args.exper_folder))
    
    config_file = 'experiment/' + args.exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
   
    config['dir'] = args.exper_folder
    trainer = Trainer(config)
    trainer.train()
    
    print('Experiment {} ended'.format(args.exper_folder))
