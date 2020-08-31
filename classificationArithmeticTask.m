result = [" Channel" "Precision" "Recall" "F-Score" ] ; % for printing out different evation metrics 
for j = 1 : 14  % loop through all the channels 
load("/home/muditdhawan/Desktop/EEG/CleanData_TDC/Ychannel" + j + "data.mat"); % loading X for a specific channel 
load("/home/muditdhawan/Desktop/EEG/CleanData_TDC/Xchannel" + j + "data.mat"); % loading labels for a specific channel 
X = X'; Y =  Y';
num = 3; % number of partitions 
indices = crossvalind('Kfold',length(X),num);  % making partitions 
avgpr = 0; avgre = 0; avgF_score = 0 ; 
for i = 1:3
    %% dividing dataset 
    test = (indices == i); 
    train = ~test;
    Xtrain= X(train,:);
    Xtest= X(test,:);
    Ytrain= Y(train,:);
    Ytest= Y(test,:);
    
    %% fitting different models 
%     Mdl = fitcsvm(Xtrain,Ytrain);  % Support vector machine (SVM) 
%     Mdl = fitckernel(Xtrain,Ytrain); % binary Gaussian kernel classification model for nonlinear classification
%      Mdl = fitcnb(Xtrain,Ytrain); % Naive Bayes 
%     Mdl = fitctree(Xtrain,Ytrain);  % clasification tree 
    Mdl = fitcensemble(Xtrain,Ytrain); % classification results of boosting 100 classification trees
    [Label,Score] = predict(Mdl,Xtest);  % Predicting 
    
    %% confusion matrix 
    cm = confusionchart(Ytest,Label);
    matr = cm.NormalizedValues;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';

    %% Precision
    pr = precision(matr);
    avgpr = avgpr + pr ;
    
    %% Recall 
    re = recall(matr);
    avgre = avgre + re ;
    
    %%% F-score
    F_score=2*re*pr/(pr+re); %%F_score=2*1/((1/Precision)+(1/Recall));
    avgF_score = avgF_score+F_score ;
    
    
    
    
end

channel = "Channel - " + j ;
result = [ result ; channel avgpr/num avgre/num avgF_score/num]; % finding avg over all the partitions 

end 
result % Result table for all the channels 
function y = precision(M)
  y = M(2,2) / (M(2,2)+ M(1,2));
end
function y = recall(M)
  y = M(2,2) / (M(2,2)+ M(2,1));
end