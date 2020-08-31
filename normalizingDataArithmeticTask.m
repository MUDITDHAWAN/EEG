for i = 1: 14 
    fileName = "/home/muditdhawan/Desktop/EEG/CleanData_TDC/Xchannel" + i + "data.mat" ;
    X = load(fileName).("X");
    mu = X - mean(X,2);
    stdu = std(X, 0, 2);
    X = (mu)./stdu ; 
    save(fileName, "X");
end
