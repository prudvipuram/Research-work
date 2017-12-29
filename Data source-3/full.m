%% Import the data
[~, ~, raw] = xlsread('input file path','Sheet1');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
input1 = reshape([raw{:}],size(raw));

%% Clear temporary variables
clearvars raw R;
%% Import the data
[~, ~, raw] = xlsread('target file path','Sheet1');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
target1 = reshape([raw{:}],size(raw));

%% Clear temporary variables
clearvars raw R;
%% Neural Networks

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
