main.m contains a deep model to classify the prognosis of COVID-19 patients between "serious" and "mild" cases from a single chest x-ray image and some personal and clinical information, as provided by the CDI database (not present here). The different methods of the class include data preparation, database adaptation and training itself.

The Code includes a Test code as well but the model needs to be trained first and then test can be executed which in return will give a confusion matrix

WARNING: it requires CSV files containig the names of the images along with the labels in the corresponding next column 

Pretrained weights from a different dataset must be added to ehnace the learning process of the ResNet Model

The code can be executed in 3 different configurations where it can be targeted only for Images, only for Clinical Images or for Combined (Images + Clinical) Data, The Desired option can be selected from code line 154 to 160

Requirements: - Python3 
              - tensorflow = 1.4
              - keras = 2.1.5

WARNING: file names of images should match the corresponding entries in the .xls file. If you plan to modify the images (e.g. for pre-processing reasons), please take care to ensure that the correspondance is kept by either keeping file names unchanged or properly updating the .xls file.

WARNING: depending on your machine, you might receive GPU out of memory errors. If that is the case, try reducing batch size (line 34 of main.py)
