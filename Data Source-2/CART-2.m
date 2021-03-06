%% Import the data
[~, ~, raw] = xlsread('file path','Foglio1','A2:M137');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
Heartfaliuredatasetpaper = table;

%% Allocate imported array to column variable names
Heartfaliuredatasetpaper.Systpressure = data(:,1);
Heartfaliuredatasetpaper.diastpressure = data(:,2);
Heartfaliuredatasetpaper.heartrate = data(:,3);
Heartfaliuredatasetpaper.weight = data(:,4);
Heartfaliuredatasetpaper.BNP = data(:,5);
Heartfaliuredatasetpaper.NYHA = data(:,6);
Heartfaliuredatasetpaper.Ejectionfraction = data(:,7);
Heartfaliuredatasetpaper.gender1male0female = data(:,8);
Heartfaliuredatasetpaper.age = data(:,9);
Heartfaliuredatasetpaper.atrialfibrillation = data(:,10);
Heartfaliuredatasetpaper.bundlebranchblock = data(:,11);
Heartfaliuredatasetpaper.ventriculartachycardia = data(:,12);
Heartfaliuredatasetpaper.y = data(:,13);

%% Clear temporary variables
clearvars data raw R;
%% Response
Y = Heartfaliuredatasetpaper.y;
disp('heart failure severity')
tabulate(Y)
% Predictor matrix
X = Heartfaliuredatasetpaper(:,1:end-1);

%% In this example, we will hold 40% of the data, selected randomly, for
% test phase.

cv = cvpartition(height(Heartfaliuredatasetpaper),'holdout',0.40);

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
