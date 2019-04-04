# Visual Recommender for Online Fashion Retailer

Project is created for study purposes. It mimics real project.


### Requirements
```
torch==1.0.1
torchvision==0.2.2
```
Script runs only on GPU. 


### Dataset

Put dataset into folder 'data/' in the root of the proejct.
Dataset used for this task is DeepFashion: In-shop Clothes Retrieval, low-resolution images were used.
Link to download: https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc

### Training and Evaluaton

Folder experiment/e002 contains config.yaml file with all the training details provided.
Command for training:
```
python trainer.py e002
```

Evaluaton script has two arguments - folder to load json file and folder with image embeddings
Command for evaluaiton:
```
python eval.py e002 data/Embed
```


### Demo

Change image_index in 'Demo.ipynb' to find similar image for diffenent image in test set.
