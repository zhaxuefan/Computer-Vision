load('traintest.mat','test_imagenames','test_labels');
load('visionRandom.mat');
%load('visionHarris.mat');
T=numel(test_imagenames);
imgPaths='G:\cmu\fall2018\computer vision\hw2\hw2\data';
filterBank=createFilterBank();
[K,~]=size(dictionary);
i=0;
testfeature=[];
for i1=1:T
        name=test_imagenames{1,i1};
        img=imread(sprintf('%s/%s', imgPaths,name));
        wordMap = getVisualWords(img, dictionary, filterBank);
        h = getImageFeatures(wordMap, K);
        testfeature=[testfeature;h];
end
%train data preprocessing
[train,pstrain] = mapminmax(trainFeatures');
pstrain.ymin = 0;
pstrain.ymax = 1;
[train,pstrain] = mapminmax(train,pstrain);
% test data preprocessing
[test,pstest] = mapminmax(testfeature');
pstest.ymin = 0;
pstest.ymax = 1;
[test,pstest] = mapminmax(test,pstest);
train = train';
test = test';
model=svmtrain(trainLabels,train,'-t 3');
%test classification
[predict_label, accuracy, dec_values]=svmpredict(test_labels',test,model);
%save visionHarrisSVM.mat dictionary filterBank trainFeatures trainLabels testfeature test_labels predict_label accuracy dec_values;