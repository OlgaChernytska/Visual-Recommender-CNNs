# Visual Recommender for Online Fashion Retailer

Project is created for study purposes. It mimics real project.


### Requirements
```
torch==1.0.1
torchvision==0.2.2
```
Script runs only on GPU. 


### Dataset

Put dataset into folder 'data/' in the root of the project. Dataset used for this task is "DeepFashion: In-shop Clothes Retrieval" (low-resolution images).

Link to download: https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc

### Training and Evaluaton

Folder experiment/e002 contains config.yaml file with all the training details provided. Model weights are provided - experiment/e002/weights_last.pth.
Command for training:
```
python trainer.py e002
```

After CNN was trained, its weights are used to create embeddings for all images in test set. Command for creating embeddings:
```
python embeddings.py e002 data/Embed
```

Evaluaton script has two arguments - folder to load json file and folder with image embeddings.
Command for evaluaiton:
```
python eval.py e002 data/Embed
```


### Demo

Demo provided in jupyter notebook. Image_index corresponds to image index in test set. Change Image_index to see model accuracy visualization for different images from test set.

Demo will work when dataset is dowloaded and embeddings are created.
