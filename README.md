# Problem Statement
To construct an **image classifier model** to distinguish between species of butterflies. The metric of interest that the model's performance is based on is **accuracy**. 

# Background
The dataset used for the project was taken from a [Kaggle dataset](https://www.kaggle.com/competitions/yum-or-yuck-butterfly-mimics-2022/overview). 

The impetus for the project stems from the fact that there are certain species of butterflies that are considered endangered or threatened. In particular, 2 of the species included in the dataset, the Monarch butterfly and the Spicebush Swallowtail, fall under this category. 

An efficient and well-performing neural network model that allows for the effective classification of the species of butterfly would allow researchers to keep tabs on the butterfly populations, while ensuring their numbers are preserved in our changing world. 

# Exploratory Data Analysis (EDA)
EDA has been performed and documented in a Jupyter Notebook, `eda.ipynb`. Below are some important observations gleaned in the process of the EDA:

<br>

# Baseline Models

## 1. Data Preparation
In this project, transfer learning of models from both [torchvision](https://pytorch.org/vision/main/models/resnet.html) and [MMClassification](https://github.com/open-mmlab/mmclassification) were used. 

In order to accomodate the folder structure to that used in the two repositories, a [data preparation script](src/data_prep.py) is written. Essentially, this script parses through the original file structure that the raw dataset uses, then performs a train-test-split of the images. Finally, the images are moved into the following file structure, depending on their train/test classification and their labels.
```
        |- data
            |- processed_data
                |- train
                    |- class_1
                        |- img_1
                        |- img_2
                        ...
                |- val
                    |- class_1
                        |- img_1
                        |- img_2
                        ...
                |- test
                    |- class_1
                        |- img_1
                        |- img_2
                        ...
```

## 2. Model Training
For GPU access (model training would take significantly longer otherwise) during model training, I have an accompanying Google Colab notebook where the models are trained. 

For the MMClassification repository, the models experimented with thus far are: 
- [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929.pdf)
- [ResNet-50](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

For the torchvision models, I have experimented with ResNet-101. 



For all model implementations, the models have been instantiated with pre-trained weights (trained on ImageNet). 

I then fine-tune the network following the approach outlined in the [cs231n notes](https://cs231n.github.io/transfer-learning/). Instead of freezing the network and replacing only the classifier head, I opted to fine-tune the weights across the pretrained network through backpropagation. 

maybe can quote this for the approach in mmclass: by freezing the earlier layers, we retain the earlier layers of the network, which have been observed to focus on more generic features (e.g. edge detectors or colour blob detectors) and instead finetune the later layers. The later layers have been found to be more specific to the details contained in the initial dataset. As ImageNet contains many dog breeds, the network may be more sensitive to featurse differentiating between dog breeds, instead of differentiating between butterfly species.




<font color='red'> **maybe write abit on why transfer learning impt** </font>

# Future Work
While the model accuracy already exceeds 90%, the finetuning of the hyperparameters of a chosen model is yet to be completed. Future development of the project would involve the resplitting of the images into train/validation/test sets, where the validation is the set that will be used for the finetuning and the test set will be used for the final assessment of the fine-tuned model. 

As postulated in the EDA, to ensure a more rigorous stratification of images into train/val/test splits, it could be worthwhile to include the side of the wing that is visible as one of the features to balance when doing the split. This would improve the representativeness of the training images, which in turn, improves the model's performance in the face of unseen data. <font color= "red"> **Maybe delete this** </font>

The arguments for the training and inference script will also be converted in a centralised config file, which will make the toggling of important arguments extremely straightforward. Experiment tracking tools (such as MLflow ) will also be able to document the arguments in the config file more easily.

Finally, MMclassification allows for the swapping of models through config files. I hope to exploit this convenience and experiment more models of various architectures on the dataset, to identify the model architecture most suited for this use case. 