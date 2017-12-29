%% Import the data
[~, ~, raw] = xlsread('file path','Sheet1','A2:N304');

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
clevelandrealdata = table;

%% Allocate imported array to column variable names
clevelandrealdata.age = data(:,1);
clevelandrealdata.sex = data(:,2);
clevelandrealdata.cp = data(:,3);
clevelandrealdata.trestbps = data(:,4);
clevelandrealdata.chol = data(:,5);
clevelandrealdata.fbs = data(:,6);
clevelandrealdata.restecg = data(:,7);
clevelandrealdata.thalach = data(:,8);
clevelandrealdata.exang = data(:,9);
clevelandrealdata.oldpeak = data(:,10);
clevelandrealdata.slope = data(:,11);
clevelandrealdata.ca = data(:,12);
clevelandrealdata.thal = data(:,13);
clevelandrealdata.y = data(:,14);

%% Clear temporary variables
clearvars data raw;

%% Response
Y = clevelandrealdata.y;
disp('heart failure severity')
tabulate(Y)
% Predictor matrix
X = clevelandrealdata(:,1:end-1);

%% In this example, we will hold 40% of the data, selected randomly, for
% test phase.

cv = cvpartition(height(clevelandrealdata),'holdout',0.40);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

disp('Training Set')
tabulate(Ytrain)
disp('Test Set')
tabulate(Ytest)
%% classification and regression tree
tic
% Train the classifier
t = ClassificationTree.fit(Xtrain,Ytrain);
toc

% Make a prediction for the test set
Y_t = t.predict(Xtest);

% Compute the confusion matrix
C_t = confusionmat(Ytest,Y_t);
% Examine the confusion matrix for each class as a percentage of the true class
C_t = bsxfun(@rdivide,C_t,sum(C_t,2)) * 100

% calclating accuracy
Cf1 = cfmatrix2(Ytest,Y_t);

view(t,'mode','graph')
view(t)

