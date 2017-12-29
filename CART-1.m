
%% Import the data
[~, ~, raw] = xlsread('file address','Sheet1','A2:P246');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
cellVectors = raw(:,16);
raw = raw(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
full = table;

%% Allocate imported array to column variable names
full.Smoking = data(:,1);
full.DustSmoke = data(:,2);
full.Dyspnea = data(:,3);
full.Age = data(:,4);
full.Sex = data(:,5);
full.RF = data(:,6);
full.InhaleT = data(:,7);
full.ExhaleT = data(:,8);
full.RFTV = data(:,9);
full.HR = data(:,10);
full.Systolic = data(:,11);
full.Diastolic = data(:,12);
full.ECGST = data(:,13);
full.Spo2 = data(:,14);
full.y = data(:,15);


%% Clear temporary variables
clearvars data raw cellVectors;

%% Response
Y = full.y;
disp('heart failure severity')
tabulate(Y)
% Predictor matrix
X = full(:,1:end-2);

%% In this example, we will hold 40% of the data, selected randomly, for
% test phase.

cv = cvpartition(height(full),'holdout',0.40);

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
