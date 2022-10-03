# Problem Statement
To construct a **multi-class image classifier model** to distinguish between species of butterflies. The metric of interest is (top-1) **accuracy**. 

# Background
The dataset used for the project was taken from a [Kaggle dataset](https://www.kaggle.com/competitions/yum-or-yuck-butterfly-mimics-2022/overview). 

The impetus for the project stems from the fact that there are certain species of butterflies that are considered endangered or threatened. In particular, 2 of the species included in the dataset, the Monarch butterfly and the Spicebush Swallowtail, fall under this category. 

An efficient and well-performing computer vision model that allows for the effective classification of the species of butterfly would allow researchers to keep tabs on the butterfly's populations, while ensuring their numbers are preserved in our changing world. 

# Exploratory Data Analysis (EDA)
EDA has been performed and documented in a [Jupyter Notebook](eda.ipynb). Below are some important observations gleaned in the process of the EDA:
- This is a multi-class image classification problem. Specifically, there are 6 distinct classes in the dataset, each representing a butterfly species of interest.
- There is a slight imbalance in the frequency of classes in the dataset. This will be accounted for through stratified sampling when spliting the images into the train, validation and test sets.
- A potentially useful feature has been provided in the form of `side`, which captures the side/ angle which the photo has been taken. This could be another feature to stratify along when performing the train-test-split, since butterflies of the same species can look different from different angles.
- Image sizes are predominantly 224x224.
- I also analysed the colour distribution for the dataset, by taking the average pixel intensities for the RGB channels of each image. 
    - Along the <font color='red'> red </font> channel, the interquartile range (IQR) of pixel intensity is generally comparable through the classes, with the exception for the viceroy class, where the range of pixel intensity is smaller. 
    - For the <font color='green'> green </font> channel, the IQR is also similar across all classes except for viceroy class, where the range is again smaller. 
    - Finally, the median pixel intensity for the <font color='cornflowerblue'>blue</font> channel is noticeably lower compared to the 2 other channels, suggesting that blue is not a prominent colour among the butterflies in the sample. 
- Analysis was also done on the brightness of the images. In particular, I derived the [relative luminance](https://en.wikipedia.org/wiki/Relative_luminance) of each image through a linear equation involving the pixels of each of the RGB channel. I found that the median brightness of each class was similar. 

For a more comprehensive walkthrough for the EDA, please refer to the EDA notebook.
<br>

# Baseline Models

## 1. Data Preparation
In this project, I used the open source image classification toolbox, [MMClassification](https://github.com/open-mmlab/mmclassification), which is designed around the PyTorch framework.

In order to accomodate the folder structure to that used in the repository, a [data preparation script](src/data_prep.py) is written. Essentially, this script parses through the original file structure that the raw dataset came with, then performs a train-test-split of the images. Finally, the images are moved into the following file structure, depending on their train/val/test classification and their labels.
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

The models experimented with thus far are: 
- [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929.pdf)
- [ResNet-50](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)


For all model implementations, the models have been instantiated with pre-trained weights (trained on ImageNet). 

I then fine-tune the network following the approach outlined in the [cs231n notes](https://cs231n.github.io/transfer-learning/). 

More specifically, I froze the earlier layers of the network (I arbitrarily set the value at 2) and finetune the weights in the rest of the network. By freezing the earlier layers, we retain the weights from ImageNet in the earlier layers of the network, which have been observed to focus on more generic features (e.g. edge detectors or colour blob detectors) and instead finetune the later layers. The later layers have been found to be more specialised in capturing the details contained in the initial dataset. However as ImageNet contains many dog breeds, the network may be more sensitive to features differentiating between dog breeds, instead of differentiating between butterfly species and hence, the finetuning of later layers is important.

## 3. Model Performance
To allow for comparability, models have been trained for 200 epochs (this can be changed through config arguments in the [train](src/train_model.py) function). Model's performance will be assessed every 10 training epochs on the validation set. The weights that gave the best validation score is then tested on the test set. Importantly, the test set was not used in the training of the model and there should be minimal data leakage. The test set accuracy is presented below:

| Model Architecture | Epochs Trained | Best Validation Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| Random Guess | - | 1/6 = 0.17 | 0.17 |
| ResNet-50 (MMCls) | 200 | 0.82 (at 170 epochs) | 0.82 |
| ViT | 200 | 0.50 (at 200 epochs) | 0.50

# Future Work
While the model accuracy for ResNet-50 already exceeds 80%, the finetuning of its hyperparameters is yet to be completed. Future development of the project would involve the further tweaking of elements such as the learning rate and the number of layers to freeze. This should further improve the performance of the model.

By default, the logging of the losses and the train/val accuracies during model training in MMClassification is in the form of print statements. I will work towards incorporating experiment tracking tools (such as MLflow), which will be able to document the evaluation metrics as the model trains in a more re-visitable and streamlined manner. 

Finally, MMClassification allows for the swapping of models through config files. I will be leveraging on this convenience and will experiment with more model architectures, to identify the model architecture most suited for this butterfly classification problem. 