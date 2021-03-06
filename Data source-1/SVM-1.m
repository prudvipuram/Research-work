%% Import the data
[~, ~, raw] = xlsread('file path','Sheet1','A2:P246');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
cellVectors = raw(:,16);
raw = raw(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
full1 = table;

%% Allocate imported array to column variable names
full1.Smoking = data(:,1);
full1.DustSmoke = data(:,2);
full1.Dyspnea = data(:,3);
full1.Age = data(:,4);
full1.Sex = data(:,5);
full1.RF = data(:,6);
full1.InhaleT = data(:,7);
full1.ExhaleT = data(:,8);
full1.RFTV = data(:,9);
full1.HR = data(:,10);
full1.Systolic = data(:,11);
full1.Diastolic = data(:,12);
full1.ECGST = data(:,13);
full1.Spo2 = data(:,14);
full1.y = data(:,15);
full1.VarName16 = cellVectors(:,1);

%% Clear temporary variables
clearvars data raw cellVectors;

%% Response
Y = full1.y;
disp('heart failure severity')
tabulate(Y)
% Predictor matrix
X = full1(:,1:end-2);

%% In this example, we will hold 40% of the data, selected randomly, for
% test phase.

cv = cvpartition(height(full1),'holdout',0.40);

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
%% support vector machine
% Train the classifier
Mdl = fitcecoc(Xtrain,Ytrain);

% Make a prediction for the test set
Y_svm = predict(Mdl,Xtest);
C_svm = confusionmat(Ytest,Y_svm);
% Examine the confusion matrix for each class as a percentage of the true class
C_svm = bsxfun(@rdivide,C_svm,sum(C_svm,2)) * 100;
% caluculate accuracy
Cf2 = cfmatrix2(Ytest,Y_svm);
