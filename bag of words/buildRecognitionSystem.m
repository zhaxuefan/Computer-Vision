% buildRecognitionSystem.m
% This script loads the visual word dictionary (in dictionaryRandom.mat or dictionaryHarris.mat) and processes
% the training images so as to build the recognition system. The result is
% stored in visionRandom.mat and visionHarris.mat.
%% save random mat 
load('traintest.mat','train_imagenames','train_labels');
T=numel(train_imagenames);
imgPaths='G:\cmu\fall2018\computer vision\hw2\hw2\data';
filterBank=createFilterBank();
load('dictionaryRandomalpha=300.mat');
[K,~]=size(dictionary);
trainFeatures=[];
for i1=1:T
        name=train_imagenames{1,i1};
        img=imread(sprintf('%s/%s', imgPaths,name));
        wordMap = getVisualWords(img, dictionary, filterBank);
        h = getImageFeatures(wordMap, K);
        trainFeatures=[trainFeatures;h];
        trainLabels=train_labels';
end
save visionRandom.mat dictionary filterBank trainFeatures trainLabels;
%% save Harris mat 
clear all;
load('traintest.mat','train_imagenames','train_labels');
T=numel(train_imagenames);
imgPaths='G:\cmu\fall2018\computer vision\hw2\hw2\data';
filterBank=createFilterBank();
load('dictionaryHarrisalpha=300.mat');
[K,~]=size(dictionary);
trainFeatures=[];
for i1=1:T
        name=train_imagenames{1,i1};
        img=imread(sprintf('%s/%s', imgPaths,name));
        wordMap = getVisualWords(img, dictionary, filterBank);
        h = getImageFeatures(wordMap, K);
        trainFeatures=[trainFeatures;h];
        trainLabels=train_labels';
end
save visionHarris_hist.mat dictionary filterBank trainFeatures trainLabels;
