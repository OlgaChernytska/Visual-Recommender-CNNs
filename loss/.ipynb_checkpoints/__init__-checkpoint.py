import torch
import torch.nn as nn
import torch.nn.functional as F 

def get_loss(config):
    loss_name = config['loss']
    
    if loss_name == 'triplet_cosine':
        return TripletLossCosine()
    else:
        raise ValueError('Unknown loss: {}'.format(loss_name))
    
       
class TripletLossCosine(nn.Module):
    def __init__(self):
        super(TripletLossCosine, self).__init__()
        self.MARGIN = 0.7
            
    def forward(self, anchor, positive, negative):
        dist_to_positive = 1 - F.cosine_similarity(anchor, positive)
        dist_to_negative = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(dist_to_positive - dist_to_negative + self.MARGIN)
        loss = loss.mean()
        return loss
   
    