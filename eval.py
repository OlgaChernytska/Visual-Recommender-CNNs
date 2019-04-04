import argparse
import yaml
import os
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
      
        
class Evaluation:
    def __init__(self, exper_folder, embed_folder):
        
        self.exper_folder = exper_folder
        self.embed_folder = embed_folder
        self.similarities, self.img_names = self._calculate_similarities()
     

    def evaluate(self):
        top_ns = [1,3,5,10]
        res = dict()
        
        for top_n in top_ns:
            top_n_accuracy = self._accuracy(top_n)
            res['top_{}_accuracy'.format(top_n)] = top_n_accuracy
    
        with open('experiment/{}/eval.json'.format(self.exper_folder), 'w') as fp:
            json.dump(res, fp) 
        
        
    def _calculate_similarities(self):
        images = os.listdir(self.embed_folder)
        images = [x for x in images if x.find('.npy')>0]
        
        embeddings = np.zeros((len(images),300))
        img_names = []
        
        for i in range(len(images)):
            npy_name = images[i]
            vector = np.load(os.path.join(self.embed_folder, npy_name))
            embeddings[i] = vector
            img_name = npy_name.replace('-','/').replace('.npy','')
            img_names.append(img_name)

        similarities = cosine_similarity(embeddings, embeddings)
        return similarities, img_names
    
    
    def _accuracy(self, top_n):
        num_total = 0
        num_correct = 0
        print('Calculating Top {} Accuracy'.format(top_n))
        
        for i in range(len(self.similarities)):
    
            if (i+1) % 1000==0:
                print('Images processed: {}. Number of correct: {}'.format(i+1, num_correct))
            
            img_name = self.img_names[i]
            this_sims = self.similarities[i]
            most_similar_ids = (-this_sims).argsort()[1:(top_n+1)]
            simiratiry_names = np.array(self.img_names)[most_similar_ids]
            res = re.match(r'.*MEN/([A-Za-z_]*/id_[0-9]*)/.*', img_name)
            num_corr = len([x for x in simiratiry_names if x.find(res.group(1))>0])
            num_total += 1
            num_correct += int(num_corr>0)
    
        top_n_accuracy = num_correct / (num_total + 0.0000001)
        print('Top {} Accuracy: {}'.format(top_n, top_n_accuracy))
        
        return top_n_accuracy
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    parser.add_argument('embed_folder', help='Provide embedding folder')
    args = parser.parse_args()
    
    print('Evaluation for experiment {}...'.format(args.exper_folder))
    
    eval_class = Evaluation(args.exper_folder, args.embed_folder)
    eval_class.evaluate()
    
    print('Evaluation done.')