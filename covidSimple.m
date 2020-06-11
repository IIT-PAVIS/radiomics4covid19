classdef covidSimple < handle
    % June 2020, Jacopo Tessadori
    % Classification on CDI dataset
    
    properties
        dataPath;
        imgData;
        setProportions=[.7,.15,.15]; % Train, validation, test
        dataSets;
        setIdxs;
        lbls;
        trndNet;
        BAcc;
        fileNames;
        patchIdx;
        patchPos;
        patchLbls;
        imgData2;
    end
    
    methods
        function this=covidSimple(path)
            % Constructor for covidSimple class.
            % Only argument (required) is absolute path of folder
            % containing images and .xls description
            this.dataPath=path;
        end
        
        function prepareDescription(this)
            % This function finds, loads and reads the .xls description
            % file in the folder, than converts it in a table. 
            
            % Find, open and read .xlsx file
            D=dir(this.dataPath);
            Dcell=struct2cell(D);
            xlsIdx=cellfun(@(x)contains(x,'xls'),Dcell(1,:));
            dataTable=readtable([this.dataPath,'\',D(xlsIdx).name]);
            
            % Some columns contain mixed numerical and 'na'. Substitute
            % 'na' with NaNs and convert to numerical
            for currCol=4:size(dataTable,2)
                relData=dataTable(:,currCol);
                cnvrt=0;
                if isa(relData{1,1},'cell')&&isa(relData{1,1}{1},'char')
                    for currRow=1:size(dataTable,1)
                        if strcmp(relData{currRow,1}{1},'na')
                           relData{currRow,1}{1}='NaN';
                           cnvrt=1;
                        end
                    end
                    if cnvrt
                        dataTable(:,currCol)=relData;
                        dataTable.(currCol)=str2double(dataTable{:,currCol});
                    end
                end
            end
            this.imgData=dataTable;
            
            % Entry 423 has no matching file, at the moment. Remove it
            this.imgData(423,:)=[];
        end
                
        function prepareImages(this,shuffleData)
            % Generate 4-layers images for further processing. Missing
            % values are (rather arbitrarily) substituted with the mean
            % value of non-missing ones, for each column. If a second
            % argument is passed and it is equal to 1, img data is
            % shuffled. If it is 2, additional data is shuffled
            
            % Switch all categorical variables to one-hot encoding and
            % substitute nans with non-nan means
            tempData=this.imgData;
            toBeRemoved=[];
            for currCol=4:size(this.imgData,2)-1
                if isa(this.imgData.(currCol),'cell') % Categorical data
                    cats=unique(tempData.(currCol));
                    nCats=length(cats);
                    tempData.(currCol)=double(double(categorical(tempData.(currCol)))==repmat(1:nCats,size(tempData,1),1))*(2^16-1);
                else % Numerical data
                    colMean=nanmean(tempData.(currCol));
                    colStd=nanstd(tempData.(currCol));
                    if isnan(colMean)||colMean==0||colStd==0
                        toBeRemoved=cat(1,toBeRemoved,currCol);
                    else
                        tempData.(currCol)(isnan(tempData.(currCol)))=nanmean(tempData.(currCol));
                        tempData.(currCol)=(tempData.(currCol)-min(tempData.(currCol)))/(max(tempData.(currCol))-min(tempData.(currCol)))*(2^16-1);
                    end
                end
            end
            tempData(:,toBeRemoved)=[];
            this.imgData=tempData;
            
            writePath=[pwd,'\Enh\'];
            if exist('Enh','dir')
                rmdir('Enh','s');
            end
            mkdir('Enh');
            
            % Add appropriate fouth-layer to images
            D=dir(this.dataPath);
            D(1:2)=[];
            Dcell=struct2cell(D);
            fNames=cell(size(tempData,1),1);
            
            % Convert all images into 4-layers TIFF and add additional info
            % to layer 4
            for currImg=1:length(fNames)
                imgVec=cellfun(@(x)contains(x,[this.imgData.(3){currImg},'_']),Dcell(1,:))|cellfun(@(x)contains(x,[this.imgData.(3){currImg},'.']),Dcell(1,:));
                fileName=Dcell{1,find(imgVec,1)};
                if sum(imgVec)>1
                    warning('Issue with file named %s. Probably another file has too similar a name.\n',fileName);
                end
                if nargin>2&&shuffleData==1 % If second argument is 1, shuffle img data
                    fNames{rndOrdr(currImg)}=fileName;
                else
                    fNames{currImg}=fileName;
                end
                img=imresize(imread([this.dataPath,'\',fileName]),[224 224]);
                addInfo=randi(10,size(img,1),size(img,2),'uint16');
                pntr=1;
                for currCol=4:size(tempData,2)-1
                    varSize=length(tempData.(currCol)(1));
                    if nargin>2&&shuffleData==2 % If second argument is 2, shuffle additional data
                        addInfo(pntr:pntr-1+varSize)=tempData.(currCol)(randi(length(fNames)));
                    else
                        addInfo(pntr:pntr-1+varSize)=tempData.(currCol)(currImg);
                    end
                    pntr=pntr+varSize;
                end
                if size(img,3)>3
                    imgTiff=img;
                    imgTiff(:,:,4)=addInfo;
                else
                    imgTiff=cat(3,img,img,img,addInfo);
                end
                imwrite(imgTiff,[writePath,fileName],'tiff');
                if mod(currImg,100)==0
                    fprintf('%d/%d\n',currImg,length(fNames));
                end
            end
            this.fileNames=fNames;
        end
        
        function prepareDatasets2Class(this,testIdx)
            % Split images in training, validation and test datasets and
            % generate corresponding imageDatastores. A second argument may
            % be passed, specifying the indeces of the images to include in
            % the test set. In that case, the setProportion property of the
            % class is ignored and 20% of the non-test samples will be used
            % as validation.
            
            % Generate lables from prognosis
            tempData=this.imgData;
            tempLbls=double(~strcmpi(tempData.Prognosi,'lieve'))+1;
            tempLbls(strcmpi(tempData.Prognosi,'altro'))=NaN;
            
            % Split images in training, test and validation datasets to
            % classify selected task
            nClasses=sum(~isnan(unique(tempLbls)));
            idxs=cell(3,1);
            idxs{1}=[];
            idxs{2}=[];
            idxs{3}=[];
            if nargin==1
                for currClass=1:nClasses
                    tempIdx=(find(tempLbls==currClass))';
                    rndIdx=randperm(length(tempIdx));
                    idxs{1}=cat(2,idxs{1},tempIdx(rndIdx(1:round(length(rndIdx)*this.setProportions(1)))));
                    idxs{2}=cat(2,idxs{2},tempIdx(rndIdx(round(length(rndIdx)*this.setProportions(1))+1:round(length(rndIdx)*this.setProportions(1))+round(length(rndIdx)*this.setProportions(2)))));
                    idxs{3}=cat(2,idxs{3},tempIdx(rndIdx(round(length(rndIdx)*this.setProportions(1))+round(length(rndIdx)*this.setProportions(2))+1:end)));
                end
            else
                idxs{3}=testIdx;
                otherIdxs=setdiff(1:length(tempLbls),idxs{3});
                rndIdx=randperm(length(otherIdxs));
                idxs{1}=otherIdxs(rndIdx(1:round(.8*length(otherIdxs))));
                idxs{2}=otherIdxs(rndIdx(round(.8*length(otherIdxs))+1:end));
                for currSet=1:3
                    idxs{currSet}(isnan(tempLbls(idxs{currSet})))=[];
                end
            end
            this.setIdxs=idxs;
            
            writePath=[pwd,'\Enh\'];
            for currSet=1:3
                currFs=matlab.io.datastore.DsFileSet(cellfun(@(x)[writePath,x],this.fileNames(idxs{currSet}),'UniformOutput',false));
                this.dataSets{currSet}=imageDatastore(currFs);
                this.lbls{currSet}=tempLbls(idxs{currSet});
                this.dataSets{currSet}.Labels=categorical(this.lbls{currSet});
                this.dataSets{currSet}=augmentedImageDatastore([224,224],this.dataSets{currSet});%,'DataAugmentation',imageAugmenter);
            end
        end
        
        function [outLbls,outEst,currLbls]=trainNet2Class(this)
            % Perform actual training of the model, than evaluate results
            % on the test set.
            if isempty(this.trndNet)
                % Trains a modified resnet18 network (changed number of output
                % classes and last FC layer size). At the moment, net is saved
                % in a file
                netFile=load('untrndNet2.mat'); % resnet18 modified with addiitional features and for generation of localization maps
                
                % Change last layer of the network to suppor different class
                % weights
                P=histcounts(this.lbls{1},'Normalization','probability');
                layers=netFile.resnet5Class.Layers;
                layers(end-2)=fullyConnectedLayer(2);
                layers(end-2).Name='fcFinal';
                layers(end)=weightedClassificationLayer(1./P);
                layers(end).Name='classoutput'; % For resnets
                lgraph=covidSimple.createLgraphUsingConnections(layers,netFile.resnet5Class.Connections);
            else
                P=histcounts(this.lbls{1},'Normalization','probability');
                layers=this.trndNet.Layers;
                layers(end)=weightedClassificationLayer(1./P);
                layers(end).Name='classoutput'; % For resnets
                layers(end).Classes='auto';
                lgraph=covidSimple.createLgraphUsingConnections(layers,this.trndNet.Connections);
            end
            
            batchSize=125;
            valFreq=ceil(this.dataSets{1}.NumObservations/batchSize);
            
            options = trainingOptions('sgdm', ...
                'Momentum',.98,...
                'ExecutionEnvironment','gpu', ...
                'MaxEpochs',1000, ...
                'MiniBatchSize',batchSize, ...
                'GradientThreshold',1, ...
                'Verbose',true, ...
                'Plots','none', ... %training-progress
                'Shuffle','every-epoch', ... % every-epoch
                'ValidationData',this.dataSets{2}, ...
                'ValidationFrequency',valFreq, ...
                'ValidationPatience',2, ...
                'InitialLearnRate',0.00005, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',15, ...
                'LearnRateDropFactor',0.9);
            this.trndNet=trainNetwork(this.dataSets{1},lgraph,options);
            
            % Perform test on test set
            BAccFun=@(Yreal,Yest)((sum((Yreal==1).*(Yest==1))/sum(Yreal==1))+(sum((Yreal==2).*(Yest==2))/sum(Yreal==2)))/2;
            outEst=this.trndNet.predict(this.dataSets{3});
            outLbls=this.trndNet.classify(this.dataSets{3});
            currLbls=this.lbls{3};
            this.BAcc=BAccFun(this.lbls{3},double(outLbls));
        end
        
        function [gradCams,testSet]=computeGradCams(this)
            % Generates a set of GRAD cam images of the current test set
            testSet=cell(this.dataSets{3}.NumObservations,1);
            for currImg=1:this.dataSets{3}.NumObservations
                testSet{currImg}=imread(this.dataSets{3}.Files{currImg});
            end
            
            nImgs=length(testSet);
            f=squeeze(activations(this.trndNet,cat(4,testSet{:}),'res5b_relu')); % Resnet
            w=this.trndNet.Layers(76).Weights;
            w=w(:,1:512);
            gradCams=cell(nImgs,2);
            for currImg=1:nImgs
                relImg=double(testSet{currImg}(:,:,1:3))/256;
                for currClass=1:size(w,1)
                    M=zeros(size(f,1),size(f,2));
                    for currW=1:size(w,2)
                        M=M+w(currClass,currW)*f(:,:,currW,currImg);
                    end
                    heatMap=imresize(M,[size(relImg,1),size(relImg,2)]);
                    gradCams{currImg,currClass}=heatMap;
                end
            end
        end
        
    end
    
    methods (Static)
        function lgraph = createLgraphUsingConnections(layers,connections)
            % lgraph = createLgraphUsingConnections(layers,connections) creates a layer
            % graph with the layers in the layer array |layers| connected by the
            % connections in |connections|.
            
            lgraph = layerGraph();
            for i = 1:numel(layers)
                lgraph = addLayers(lgraph,layers(i));
            end
            
            for c = 1:size(connections,1)
                lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
            end
        end
    end
end