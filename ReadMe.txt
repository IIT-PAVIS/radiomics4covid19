main.m contains a deep model to classify the prognosis of COVID-19 patients between "serious" and "mild" cases from a single chest x-ray image and some personal and clinical information, as provided by the CDI database (not present here). The different methods of the class include data preparation, database adaptation and training itself.

The Code includes a Test code as well but the model needs to be trained first and then test can be executed which in return will give a confusion matrix

WARNING: it requires CSV files containig the names of the images along with the labels in the corresponding next column 

Pretrained weights from a different dataset must be added to ehnace the learning process of the ResNet Model. To help the pretrained weights are uploaded in split arcive which can be downloaded and then used. 
The dataset used to pretrain the weights is public and is accessable on the given link (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The code can be executed in 3 different configurations where it can be targeted only for Images, only for Clinical Images or for Combined (Images + Clinical) Data, The Desired option can be selected from code line 154 to 160

Requirements: - Python3 
              - tensorflow = 1.4
              - keras = 2.1.5

WARNING: file names of images should match the corresponding entries in the .xls file. If you plan to modify the images (e.g. for pre-processing reasons), please take care to ensure that the correspondance is kept by either keeping file names unchanged or properly updating the .xls file.

WARNING: depending on your machine, you might receive GPU out of memory errors. If that is the case, try reducing batch size (line 33 of main.py)

Database, algorithm and alternative approaches are described in detail in the paper "AIforCOVID: predicting the clinical outcomes in patients with COVID-19 applying AI to chest-X-rays. An Italian multicentre study", available at https://arxiv.org/abs/2012.06531

Database is available upon request at https://aiforcovid.radiomica.it/
