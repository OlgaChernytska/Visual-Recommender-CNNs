import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    
def show_batch(batch, max_img_num=5):
    
    def _denormalize_image(img):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array(NORMALIZATION_MEAN)
        std = np.array(NORMALIZATION_STD)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img
    
    
    plt.figure(figsize=[10,18])
    img_num = min(max_img_num, len(batch['name']))
    anchors = batch['anchor']
    positives = batch['positive']
    negatives = batch['negative']
    
    for i in range(img_num):
        
        anchor = anchors[i]
        img = _denormalize_image(anchor)
        plt.subplot(img_num,3,3*i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Anchor {}'.format(i+1))
        
        positive = positives[i]
        img = _denormalize_image(positive)
        plt.subplot(img_num,3,3*i+2)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Positive {}'.format(i+1))
        
        negative = negatives[i]
        img = _denormalize_image(negative)
        plt.subplot(img_num,3,3*i+3)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Negative {}'.format(i+1))

        
    plt.show()
    return
        
        
def get_embeddings(folder):
    images = os.listdir(folder)
    images = [x for x in images if x.find('.npy')>0]
    embeddings = np.zeros((len(images),300))
    img_names = []

    for i in range(len(images)):
        npy_name = images[i]
        vector = np.load(os.path.join(folder, npy_name))

        embeddings[i] = vector
        img_name = npy_name.replace('-','/').replace('.npy','')
        img_names.append(img_name)
        
    print('Embeddings shape: {}'.format(embeddings.shape))
    return embeddings, img_names


def show_similar_images(target_id, img_names, similarities):
    img_name = img_names[target_id]
    this_sims = similarities[target_id]
    most_similar_ids = (-this_sims).argsort()[:12]
    similarity_values = this_sims[most_similar_ids]
    simiratiry_names = np.array(img_names)[most_similar_ids]
    plt.figure(figsize=[15,13])
    
    for i, name in enumerate(simiratiry_names):
        img = Image.open('data/'+name)
        res = re.match(r'.*MEN/([A-Za-z_]*)/(id_[0-9]*)/.*', name)
        category = res.group(1)
        product = res.group(2)
        plt.subplot(3,4,i+1)
        plt.imshow(img)
        plt.axis('off')
        if i==0:
            plt.title('Target Image\n Category: {}\n Product: {}'
                  .format(category, product))
        else:
            plt.title('Similarity: {:.3f}\n Category: {}\n Product: {}'
                  .format(similarity_values[i], category, product))