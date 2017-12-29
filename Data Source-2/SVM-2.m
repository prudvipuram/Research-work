%% Import the data
[~, ~, raw] = xlsread('file path','Sheet1','A2:M137');

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
hfseverity = table;

%% Allocate imported array to column variable names
hfseverity.PA_S = data(:,1);
hfseverity.PA_D = data(:,2);
hfseverity.HR = data(:,3);
hfseverity.WEIGHT = data(:,4);
hfseverity.BNP = data(:,5);
hfseverity.NYHA = data(:,6);
hfseverity.EF = data(:,7);
hfseverity.GENDER = data(:,8);
hfseverity.AGE = data(:,9);
hfseverity.FibAtr = data(:,10);
hfseverity.Blocbran = data(:,11);
hfseverity.tacventr = data(:,12);
hfseverity.y = data(:,13);

%% Clear temporary variables
clearvars data raw;
%% Response
Y = hfseverity.y;
disp('heart failure severity')
tabulate(Y)
% Predictor matrix
X = hfseverity(:,1:end-1);

%% In this example, we will hold 40% of the data, selected randomly, for
% test phase.

cv = cvpartition(height(hfseverity),'holdout',0.40);

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
