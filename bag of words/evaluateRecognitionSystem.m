% evaluateRecognitionSystem.m
% This script evaluates nearest neighbour recognition system on test images
% load traintest.mat and classify each of the test_imagenames files.
% Report both accuracy and confusion matrix
load('traintest.mat','test_imagenames','test_labels');
load('visionRandom.mat');
%load('visionHarris_hist.mat');
T=numel(test_imagenames);
imgPaths='G:\cmu\fall2018\computer vision\hw2\hw2\data';
filterBank=createFilterBank();
[K,~]=size(dictionary);
method='euclidean';
%method='chi2';
i=0;
testlabel=[];
for i1=1:T
        name=test_imagenames{1,i1};
        img=imread(sprintf('%s/%s', imgPaths,name));
        wordMap = getVisualWords(img, dictionary, filterBank);
        h = getImageFeatures(wordMap, K);
        dist= getImageDistance(h, trainFeatures, method);
        [~,label]=min(dist);
        testlabel=[testlabel;trainLabels(label)];
end
i=nnz(testlabel'-test_labels);
accuracy=i/T;
C=confusionmat(test_labels,testlabel');