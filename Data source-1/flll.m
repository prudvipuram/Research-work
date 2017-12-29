%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: C:\Users\Shravya\Documents\independent stdy\dataset\dataset teja\neural networks\full\input.xlsx
%    Worksheet: Sheet1
%
% To extend the code for use with different selected data or a different
% spreadsheet, generate a function instead of a script.

% Auto-generated by MATLAB on 2016/11/15 13:59:41

%% Import the data
[~, ~, raw] = xlsread('C:\Users\Shravya\Documents\independent stdy\dataset\dataset teja\neural networks\full\input.xlsx','Sheet1');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
input = reshape([raw{:}],size(raw));
%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: C:\Users\Shravya\Documents\independent stdy\dataset\dataset teja\neural networks\full\target.xlsx
%    Worksheet: Sheet1
%
% To extend the code for use with different selected data or a different
% spreadsheet, generate a function instead of a script.

% Auto-generated by MATLAB on 2016/11/15 14:01:41

%% Import the data
[~, ~, raw] = xlsread('C:\Users\Shravya\Documents\independent stdy\dataset\dataset teja\neural networks\full\target.xlsx','Sheet1');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
target = reshape([raw{:}],size(raw));

%% Clear temporary variables
clearvars raw R;

%% Clear temporary variables
clearvars raw R;
x = input;
t = target;

net = feedforwardnet(8)

net.divideParam.trainRatio=.7;
net.divideParam.valRatio=.15;
net.divideParam.testRatio=.15;

[net,tr] = train(net,x,t);

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(testX)

performance = mse(net,testT,testY)

Outputs = sim(net,x);
Y_nn = round(testY);
view(net)

C_nn = confusionmat(testT,Y_nn)

% calculate model accuracy

Cf3 = cfmatrix2(testT,Y_nn);