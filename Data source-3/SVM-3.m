%% Import the data
[~, ~, raw] = xlsread('file path','Sheet1','A2:N304');

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
NORMALISEDCLEVELANDDATA = table;

%% Allocate imported array to column variable names
NORMALISEDCLEVELANDDATA.AGE = data(:,1);
NORMALISEDCLEVELANDDATA.sex = data(:,2);
NORMALISEDCLEVELANDDATA.CP = data(:,3);
NORMALISEDCLEVELANDDATA.TRESTBPS = data(:,4);
NORMALISEDCLEVELANDDATA.CHOL = data(:,5);
NORMALISEDCLEVELANDDATA.fbs = data(:,6);
NORMALISEDCLEVELANDDATA.RESTECG = data(:,7);
NORMALISEDCLEVELANDDATA.THLACH = data(:,8);
NORMALISEDCLEVELANDDATA.EXANG = data(:,9);
NORMALISEDCLEVELANDDATA.OLDPEAK = data(:,10);
NORMALISEDCLEVELANDDATA.SLOPE = data(:,11);
NORMALISEDCLEVELANDDATA.CA = data(:,12);
NORMALISEDCLEVELANDDATA.THAL = data(:,13);
NORMALISEDCLEVELANDDATA.y = data(:,14);

%% Clear temporary variables
clearvars data raw;
%% Response
Y = NORMALISEDCLEVELANDDATA.y;
disp('heart failure severity')
tabulate(Y)
% Predictor matrix
X = NORMALISEDCLEVELANDDATA(:,1:end-1);

%% In this example, we will hold 40% of the data, selected randomly, for
% test phase.

cv = cvpartition(height(NORMALISEDCLEVELANDDATA),'holdout',0.40);

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
