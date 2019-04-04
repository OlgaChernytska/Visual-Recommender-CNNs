import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from utils import NORMALIZATION_MEAN, NORMALIZATION_STD


class DeepFashion(Dataset):
    '''
    PyTorch class for DeepFashion: In-shop Clothes Retrieval
    Link: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html
       
    Data issues mentioned:
    - test/val split is incorrect - photos for one product are in different sets
    - some bboxes are incorrect
    '''
    
    
    def __init__(self, config):
        
        '''config is dict with required keys - path, augment, scope'''
        
        self.root_dir = config['path']
        self.scope = config['scope']

        self.image_data = train_test_split(self.root_dir, self.scope)
        self.bbox_data = bbox_annotations(self.root_dir)
  
        if self.scope == 'train':
            self.augment = bool(config['augment'])
        else:
            self.augment = False
       
        self.image_size = (256,256) # (width, height)
        
        
        if self.augment:
            self.transform = transforms.Compose([
                           transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.Resize((self.image_size[1], self.image_size[0])),
                           transforms.ToTensor(),
                           transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
                           ])
                  
        else: 
            self.transform = transforms.Compose([
                           transforms.Resize((self.image_size[1], self.image_size[0])),
                           transforms.ToTensor(),
                           transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
                           ])
            
        print('Images: {}. Augmentation: {}. Scope: {}'.format(len(self.image_data), 
                                                               self.augment, self.scope))
        
    
    def __getitem__(self, idx):
        
        '''
        for train and validation triplets are required, for prediction - only images;
        images are cropped based on bboxes
        '''
           
        row = self.image_data.iloc[idx]
        anchor_name = row['image_name']
        anchor = Image.open(os.path.join(self.root_dir, anchor_name))
        anchor = anchor.crop(get_image_bbox(anchor_name, self.bbox_data))
        anchor = self.transform(anchor)
    
        if (self.scope == 'train') or (self.scope == 'val'):
            anchor_prod_id = row['item_id']
            anchor_prod_gender = row['gender']
            anchor_prod_category = row['category']

            positive_candidates = list(self.image_data[self.image_data['item_id']==anchor_prod_id]['image_name'])
            positive_candidates = [x for x in positive_candidates if x!=anchor_name]
        
            if len(positive_candidates)==0:
                positive_name = anchor_name
            else:
                positive_name = np.random.choice(positive_candidates)
        
            negative_candidates = list(self.image_data[(self.image_data['gender']==anchor_prod_gender) & 
                         (self.image_data['category']==anchor_prod_category) & 
                         (self.image_data['item_id']!=anchor_prod_id)]['image_name'])
            negative_name = np.random.choice(negative_candidates)

            
            positive = Image.open(os.path.join(self.root_dir, positive_name))
            negative = Image.open(os.path.join(self.root_dir, negative_name)) 
    
            positive = positive.crop(get_image_bbox(positive_name, self.bbox_data))
            negative = negative.crop(get_image_bbox(negative_name, self.bbox_data))
        
            positive = self.transform(positive)
            negative = self.transform(negative)
    
        
            return {'name': anchor_name,
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                   }
        
        else:
            return {'name': anchor_name,
                    'anchor': anchor}
    

    def __len__(self):
        return len(self.image_data)



def train_test_split(root_dir, scope): 
    fn = os.path.join(root_dir, 'Eval/list_eval_partition.txt')
    df = pd.read_table(fn, skiprows=2, header=None)
    
    df['image_name'] = df[0].str.extract(r'(img/.*\.jpg)')
    df['item_id'] = df[0].str.extract(r' (id_.*)')
    df['evaluation_status'] = df['item_id'].str.split(' ', expand=True)[1]
    df['item_id'] = df['item_id'].str.split(' ', expand=True)[0]
    df = df[['image_name','item_id','evaluation_status']]
    df['gender'] = df['image_name'].str.extract(r'img/(.*MEN)/.*')
    df['category'] = df['image_name'].str.extract(r'img/.*MEN/(.*)/id_.*')
  
    if scope == 'train':
        df = df[df['evaluation_status']=='train']
    elif scope == 'val':
        df = df[df['evaluation_status']=='gallery']
    else:
        df = df[df['evaluation_status']=='query']
    return df


def bbox_annotations(root_dir):
    anno_file = os.path.join(root_dir, 'Anno/list_bbox_inshop.txt')
    df = pd.read_table(anno_file, skiprows=2, header=None)
    df = df[0].str.extract(r'(img/.*\.jpg) *([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)')
    df.columns = ['image_name', 'clothes_type', 'pose_type', 'x_1', 'y_1', 'x_2', 'y_2']
    df = df[['image_name','x_1','y_1','x_2','y_2']]
    
    return df


def get_image_bbox(img_name, df_anno):
    bbox = df_anno[df_anno['image_name']==img_name][['x_1','y_1','x_2','y_2']]
    bbox = [int(x) for x in bbox.values[0]]
    return bbox

