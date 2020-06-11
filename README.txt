covidSimple.m describes a class to train a deep model to classify the prognosis of COVID-19 patients between "serious" and "mild" cases from a single chest x-ray image and some personal and clinical information, as provided by the CDI database (not present here). The different methods of the class include data preparation, database adaptation and training itself.

covidSimpleTest.m is a script that implements an example in which several repetitions of a N-fold classification task can be  executed.
WARNING: it requires files present in this repository to be unzipped in a Matlab path folder. It will prompt the user to do so on first execution.

preTrainModels.zip.XXX are the two pre-trained models needed to run the example in covidSimpleTest, while weightedClassificationLayer.m describes a custom classification layer that allows different classes to have different weights and misclassification errors.

Requirements: - Matlab Deep Learning Toolbox
              - Unzipping tool capable of handling multi-part archives (tested with Winrar and 7-zip File Manager)
              
WARNING: file names of images should match the corresponding entries in the .xls file. If you plan to modify the images (e.g. for pre-processing reasons), please take care to ensure that the correspondance is kept by either keeping file names unchanged or properly updating the .xls file.
