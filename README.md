# A model for classifying Nuts, Bolts & more...

## Table of Contents

+ [Introduction](#introduction)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [General Layout](#general_layout)
+ [How to run](#how_to_run)


## Introduction <a name = "introduction"></a>
The idea of this project is to identify these objects using regular (conventional) image processing algorithms, and identify the objects using deep learning. To see the advantages and disadvantages of both.

![image of all parts](./image_objects_ordered/val_image.jpg)

As you can see there are 19 different objects we need to identify. We chose for 5 classes with the following information per class. 

### Different classes:

1. Nuts
    - Position
    - Oriëntation
2. Bolts
    - Position
    - Oriëntation
    - Size
    - Color
    - Type
3. Ring
    - Position
    - Size
    - Color
4. Check Valve (Festo)
    - Position
    - Oriëntation
    - Size
5. Metal Attachment
    - Position
    - Oriëntation

## Getting Started <a name = "getting_started"></a>

### Prerequisites  <a name = "prerequisites"></a>

for all the Prerequisites, I would suggest downloading all the requirements.txt file with:

```ShellSession

$ pip install -r requirements.txt
```

if there are any problems please open a ticket or send me a message *(The requirements are made with pipreqs)*.

## General Layout <a name = "general_layout"></a>

```bash
├── conventional
│   ├── 'input_conventional' <- here are the input images for conventional image processing
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── 'labeled_images' <- these are images for contour detection inside conventional.py
│   │   ├── bolt.jpg
│   │   └── nut.jpg
│   ├── 'output_conventional' <- these are the output images for conventional image processing
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── Conventional.py <- file to run for identifying objects and there features using conventional methods
│   └── output_data.csv <- output file where you can find data of every contour in the images
├── deep_learning
│   ├── '__pycache__' <- custom config files, should not  have to touch these
│   │   ├── classification.cpython-38.pyc
│   │   └── ...
│   ├── 'output' 
│   │   ├── events.out.tfevents.1674069579.LAPTOP-3VKG848S.19124.0
│   │   ├── last_checkpoint
│   │   ├── metrics.json
│   │   └── model_final.pth <- Our weigths for classification, verry important
│   ├── 'output_test'
│   │   ├── Image0-labeled.jpg
│   │   └── ...
│   ├── 'test' <- folder with files to test our alghoritm
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── 'train' <- training folder, here should be the bulk of our images
│   │   ├── annotations.json <- has annotations for our project. can be opened with vgg image annotator 
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── 'valid' <- map for validation
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── custom_config_detectron2.py <- the custom configuration file
│   ├── det_clas.py 
│   ├── environment.yml 
│   ├── output_data_det.csv <- file with output data
│   └── test_notebook.ipynb
├── 'image_objects_ordered' <- just an image for the README.md (can be used for validation)
│   ├── conding.txt 
│   └── val_image.jpg
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

By the time of this writing we are still making the deep learning algoritm (Using Detectron 2).

## How to run <a name = "how_to_run"></a>

if you've installed everything correctly you should be able to run the application with:

```ShellSession

$ python ./Conventional.py
```

or 

```ShellSession

$ python det_pred.py
```

Depending on your preference. 

**NOTE**: We expect you to run these commands in the home dir of the repo.