clear
close all
clc

% This simple code trains a modified resnet to perform prognosis of
% confirmed COVID cases between "mild" and "severe" cases. Here, a COVID
% case is defined as "severe" if it required hospitalization, intubation or
% led to subject death.

% Define some variables
nReps=25; % Number of repetitions executed
nFolds=5; % n-fold cross validation is performed

% Generate an instance of the covidSimple class. The only argument required
% is the path of the folder containing images and their .xls description.
% If the default directory exists, it will be used, otherwise prompts the
% user to select database position
defaultPath='D:\Data\2020_05_CDI_sampleFixed';
if exist(defaultPath,'dir')
    covidTest=covidSimple(defaultPath);
else
    covidTest=covidSimple(uigetdir(pwd,'Select folder containing CDI dataset:'));
end

% Look for pre-trained model files to be used in the following of this
% example. If they cannot be found but the zip files can, unzip them. If
% neither can be found, return an error and exit
if ~exist('preTrainedNet.mat','file')
    if exist('preTrainedModelsl.zip.001','file')
        unzip('preTrainedModelsl.zip.001');
    else
        error('Cannot find files for preTrained models or their compressed version (''preTrainedNet.mat'' and ''preTrainedModelsl.zip.00X''');
    end
end

% Modifies entries in the database in order to make them usable: missing
% values are (rather arbitrarily) substituted with the mean of present
% ones, while categorical variables are change to a one-hot encoding
% scheme. Entries with unkown result are removed
covidTest.prepareDescription;

% Additional information (gender, age, comoborbidities...) of patients are
% added as a fourth layer to the images, in order to be processed directly
% by the modified ResNet
covidTest.prepareImages;

% Two different conditions are tested: in one case the starting network
% has been pre-trained on the imageNet dataset, in the second the starting
% network has been pre-trained on a different classification task on
% another dataset of chest x-rays.
outLbls=cell(2,1);
outEst=cell(2,1);
lbls=cell(2,1);
for currCase=1:2
    fprintf('Case %d of 2\n',currCase);
    outLbls{currCase}=zeros(size(covidTest.imgData,1),nReps,1);
    outEst{currCase}=zeros(size(covidTest.imgData,1),nReps,2);
    lbls{currCase}=zeros(size(covidTest.imgData,1),nReps,1);
    testSetSize=round(size(covidTest.imgData,1)/nFolds);
    switch currCase
        case 1
            for currRep=1:nReps
                rndIdx=randperm(size(covidTest.imgData,1));
                for currFold=1:nFolds
                    covidTest.trndNet=[];
                    relIdx=rndIdx((currFold-1)*testSetSize+1:currFold*testSetSize);
                    covidTest.prepareDatasets2Class(relIdx);
                    relIdx=covidTest.setIdxs{3}; % Some entries might have been removed
                    [outLbls{currCase}(relIdx,currRep),outEst{currCase}(relIdx,currRep,:),lbls{currCase}(relIdx,currRep)]=covidTest.trainNet2Class;
                end
                fprintf('%d/%d\n',currRep,nReps);
            end
        case 2
            inData=load('preTrainedNet');
            for currRep=1:nReps
                rndIdx=randperm(size(covidTest.imgData,1));
                for currFold=1:nFolds
                    relIdx=rndIdx((currFold-1)*testSetSize+1:currFold*testSetSize);
                    covidTest.prepareDatasets2Class(relIdx);
                    relIdx=covidTest.setIdxs{3}; % Some entries might have been removed
                    covidTest.trndNet=inData.resnet5Class;
                    [outLbls{currCase}(relIdx,currRep),outEst{currCase}(relIdx,currRep,:),lbls{currCase}(relIdx,currRep)]=covidTest.trainNet2Class;
                end
                fprintf('%d/%d\n',currRep,nReps);
            end
    end
end

% Compute and plot accuracy results
BAccFun=@(Yreal,Yest)((sum((Yreal==1).*(Yest==1))/sum(Yreal==1))+(sum((Yreal==2).*(Yest==2))/sum(Yreal==2)))/2;
BAccs=zeros(length(outLbls),size(outLbls{1},2));
for currSet=1:size(BAccs,1)
    for currRep=1:size(BAccs,2)
        BAccs(currSet,currRep)=BAccFun(lbls{currSet}(:,currRep),outLbls{currSet}(:,currRep));
    end
end
boxplot(BAccs','labels',{'ImageNet preTr','CheXpert preTr'})
ylabel('Balanced accuracy');
title('Classification results')
h=findall(gcf,'Type','Line');
set(h,'LineWidth',1.5)
set(gca,'LineWidth',1.5,'TickDir','out')

% Plot confusion matrix and ROC curve
figure;
CM=confusionmat(lbls{2}(lbls{2}(:,3)~=0,3),outLbls{2}(lbls{2}(:,3)~=0,3));
confusionchart(CM,{'Mild','Severe'},'ColumnSummary','column-normalized','RowSummary','row-normalized')
title('Clsasification confusion matrix');
figure;
[X,Y,T,AUC]=perfcurve(double(lbls{2}(lbls{2}(:,3)~=0,3)),double(outEst{2}(lbls{2}(:,3)~=0,3))+1,1);
plot(X,Y)
h=findall(gcf,'Type','Line');
set(h,'LineWidth',1.5)
set(gca,'LineWidth',1.5,'TickDir','out')
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC');
text(.5,.5,sprintf('AUC = %0.2f',AUC))

% Compute GRAD cams and display some examples
[gradMaps,testSet]=covidTest.computeGradCams;
classEst=covidTest.trndNet.predict(cat(4,testSet{:}));
[~,imgOrdr]=sort(classEst(:,1));
imgsPerRow=4;
for currSet=1:1
    figure;
    for currImg=1:imgsPerRow
        subplot('position',[(currImg-1)/imgsPerRow,.5,1/imgsPerRow,.5]);
        CAMshow(testSet{imgOrdr((currSet-1)*imgsPerRow+currImg)}(:,:,1),gradMaps{imgOrdr((currSet-1)*imgsPerRow+currImg),2});
        set(gca,'XTickLabel',[],'YTIckLabel',[]);
        subplot('position',[(currImg-1)/imgsPerRow,0,1/imgsPerRow,.5]);
        CAMshow(testSet{imgOrdr(end-((currSet-1)*imgsPerRow+currImg))}(:,:,1),gradMaps{imgOrdr(end-((currSet-1)*imgsPerRow+currImg)),1});
        set(gca,'XTickLabel',[],'YTIckLabel',[]);
    end
end